"""Train and test the news model: a text model that reads headlines, feeding a price model trained on
what stocks actually did after them.

  text model   ridge regression from the headline's meaning vector + event flags to the abnormal move
               (out-of-fold predictions on the training years, so the price model never sees a text
               prediction made with its own label)
  price model  LightGBM from the text model's prediction, FinBERT, event flags, novelty and the market
               context at entry to the abnormal move

Separate models for intraday headlines (label: next 60 minutes) and overnight ones (label: 09:31 to the
close). Train up to 2025-11-13, tune on the next six months, final test from 2026-05-14 (never used
for any choice).

Run:  python -m backend.research.news_ml_train
"""
import json
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

from backend import config
from backend.research.news_ml_text import EVENTS

D = config.DATA_DIR / "newsml"
TRAIN_END = pd.Timestamp("2025-11-14", tz="America/New_York")
TEST_START = pd.Timestamp("2026-05-14", tz="America/New_York")
LABEL = {"intraday": "y60", "overnight": "yclose"}
CLIP = 300   # bps: one extreme move should not dominate the fit
EV = [f"ev_{k}" for k in EVENTS]
CONTEXT = ["minute", "gap", "spy_gap", "pre5", "pre30", "day_so_far", "spy_pre30", "spy_day", "rvol30", "beta",
           "log_price", "big30", "n_symbols", "novelty_sim", "news_24h", "finbert"]
FEATURES = ["text_pred"] + EV + CONTEXT


def cost_bps(d):
    return np.where(d["big30"] == 1, 10.0, 25.0)   # round trip: large caps vs the rest


def text_matrix(d, V):
    return np.hstack([V[d["row"].to_numpy()], d[EV].to_numpy(np.float32)])


def fit_text(X, y, alpha=30.0):
    return Ridge(alpha=alpha).fit(X, y)


def oof_text(d, V, y, folds=5):
    """Out-of-fold text predictions over the training years, in time blocks."""
    order = np.argsort(d["time"].to_numpy()); blocks = np.array_split(order, folds)
    X = text_matrix(d, V); pred = np.zeros(len(d))
    for b in blocks:
        mask = np.ones(len(d), bool); mask[b] = False
        pred[b] = fit_text(X[mask], y[mask]).predict(X[b])
    return pred


def trades(d, pred, keep_frac):
    """Trade the strongest predictions: one per ticker per hour (intraday) or per session (overnight)."""
    x = d.assign(pred=pred)
    if x["kind"].iloc[0] == "overnight":
        x = x.groupby(["symbol", "session"]).agg(pred=("pred", "mean"), y=("y", "first"), big30=("big30", "first"),
                                                 session=("session", "first")).reset_index(drop=True)
    else:
        x = x.sort_values("time"); x["hour"] = x["time"].dt.floor("60min")
        x = x.groupby(["symbol", "hour"]).first().reset_index()
    cut = np.quantile(np.abs(x["pred"]), 1 - keep_frac)
    t = x[np.abs(x["pred"]) >= cut]
    net = np.sign(t["pred"]) * t["y"] - cost_bps(t)
    days = x["session"].nunique()
    return {"trades": len(t), "per_day": round(len(t) / max(days, 1), 2), "net_bps": round(net.mean(), 1),
            "t": round(net.mean() / net.std() * np.sqrt(len(net)), 2) if len(net) > 2 else np.nan,
            "win": round((net > 0).mean(), 3)}


def main():
    d = pd.read_parquet(D / "articles_text.parquet").reset_index(drop=True)
    V = np.load(D / "vectors.npy")
    d["row"] = np.arange(len(d))
    d["log_price"] = np.log(d["price"]); d["big30"] = d["symbol"].isin(config.TRAIN_TICKERS).astype(int)
    report, saved = {}, {}
    for kind, lab in LABEL.items():
        k = d[(d["kind"] == kind) & d[lab].notna()].copy()
        k["y"] = k[lab]; yc = k["y"].clip(-CLIP, CLIP).to_numpy()
        tr, va, te = (k["time"] < TRAIN_END).to_numpy(), ((k["time"] >= TRAIN_END) & (k["time"] < TEST_START)).to_numpy(), (k["time"] >= TEST_START).to_numpy()
        # text model: out-of-fold on train, fitted on all of train for the later periods
        k["text_pred"] = 0.0
        k.loc[tr, "text_pred"] = oof_text(k[tr], V, yc[tr])
        text = fit_text(text_matrix(k[tr], V), yc[tr])
        k.loc[~tr, "text_pred"] = text.predict(text_matrix(k[~tr], V))
        # price model on top
        params = dict(objective="huber", alpha=60.0, learning_rate=0.03, num_leaves=15, min_data_in_leaf=200,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=5.0, verbose=-1)
        dtr = lgb.Dataset(k.loc[tr, FEATURES], yc[tr]); dva = lgb.Dataset(k.loc[va, FEATURES], yc[va])
        gbm = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
        rows = {}
        for part, m in (("tune", va), ("test", te)):
            p_all = gbm.predict(k.loc[m, FEATURES], num_iteration=gbm.best_iteration)
            sub = k[m]
            rows[part] = {
                "n": int(m.sum()),
                "rank_corr_combined": round(spearmanr(p_all, sub["y"])[0], 4),
                "rank_corr_text_only": round(spearmanr(sub["text_pred"], sub["y"])[0], 4),
                "rank_corr_finbert": round(spearmanr(sub["finbert"].fillna(0), sub["y"])[0], 4),
                "top20%": trades(sub, p_all, 0.2), "top5%": trades(sub, p_all, 0.05),
                "finbert_top20%": trades(sub, sub["finbert"].fillna(0).to_numpy(), 0.2),
            }
        imp = pd.Series(gbm.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
        rows["top_features"] = {f: round(v / imp.sum(), 3) for f, v in imp.head(10).items()}
        rows["trees"] = gbm.best_iteration
        report[kind] = rows
        saved[kind] = {"text": text, "gbm": gbm.model_to_string(num_iteration=gbm.best_iteration), "features": FEATURES}
    joblib.dump(saved, D / "news_model.joblib")
    (D / "news_model_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
