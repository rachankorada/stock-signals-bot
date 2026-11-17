"""Correctness checks for the parts where a silent bug would fake good results."""
import numpy as np
import pandas as pd
import pytest

from backend import config, db
from backend.backtest.simulator import make_levels, simulate
from backend.data.bars import resample
from backend.engine.core import Engine
from backend.features.build import FEATURES, compute_features, directional
from backend.strategies.library import find_setups

ET = "America/New_York"


@pytest.fixture(autouse=True)
def fixed_geometry(monkeypatch):
    """The expected numbers below assume these levels, whatever the live settings are."""
    for name, value in (("STOP_MIN_ATR", 0.8), ("STOP_MAX_ATR", 2.0), ("TP1_R", 1.5), ("TP2_R", 2.5)):
        monkeypatch.setattr(config, name, value)


def make_bars(days=30, seed=1) -> pd.DataFrame:
    """Random-walk 1-minute bars for full sessions."""
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex([])
    for day in pd.bdate_range("2025-03-03", periods=days):
        idx = idx.append(pd.date_range(day + pd.Timedelta(minutes=570), periods=390, freq="1min", tz=ET))
    close = 100 + np.cumsum(rng.normal(0, 0.05, len(idx)))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.normal(0, 0.03, len(idx)))
    return pd.DataFrame({
        "open": open_, "high": np.maximum(open_, close) + spread, "low": np.minimum(open_, close) - spread,
        "close": close, "volume": rng.integers(1_000, 50_000, len(idx)).astype(float),
    }, index=idx)


def test_resample_aggregates_by_session_bucket():
    bars = make_bars(days=2)
    five = resample(bars, 5)
    assert len(five) == 2 * 78
    first = bars.iloc[:5]
    assert five.iloc[0]["open"] == first["open"].iloc[0]
    assert five.iloc[0]["high"] == first["high"].max()
    assert five.iloc[0]["close"] == first["close"].iloc[-1]
    assert five.iloc[0]["volume"] == first["volume"].sum()
    assert five.index[78].hour == 9 and five.index[78].minute == 30  # buckets restart each day


@pytest.mark.parametrize("tf", [1, 5, 15])
def test_features_use_no_future_data(tf):
    """A bar's features must not change when later bars are added."""
    bars = resample(make_bars(), tf)
    cut = len(bars) - 200
    full = compute_features(bars, tf)[FEATURES].iloc[cut - 1]
    partial = compute_features(bars.iloc[:cut], tf)[FEATURES].iloc[-1]
    pd.testing.assert_series_equal(full, partial, check_names=False, rtol=1e-9)


def test_levels_respect_atr_bounds_and_direction():
    stop, tp1, tp2, risk = make_levels(100.0, np.array([99.99, 90.0, 101.0]), 1.0, np.array([1, 1, 1]))
    assert risk[0] == config.STOP_MIN_ATR      # too tight -> widened
    assert risk[1] == config.STOP_MAX_ATR      # too wide -> capped
    assert risk[2] == config.STOP_MIN_ATR      # stop on the wrong side -> minimum
    assert (stop < 100).all() and (tp1 > 100).all() and (tp2 > tp1).all()
    short_stop, short_tp1, _, _ = make_levels(100.0, np.array([101.0]), 1.0, np.array([-1]))
    assert short_stop[0] == 101.0 and short_tp1[0] == 100 - config.TP1_R


def _flat_frame(n=60) -> pd.DataFrame:
    idx = pd.date_range("2025-03-03 10:00", periods=n, freq="1min", tz=ET)
    f = pd.DataFrame({"open": 100.0, "high": 100.05, "low": 99.95, "close": 100.0, "atr": 1.0}, index=idx)
    f["mins"] = (idx.hour * 60 + idx.minute - 570) + 1
    for col in FEATURES:
        if col not in f:
            f[col] = 0.0
    return f


def _one_setup(direction=1):
    stop_ref = 99.0 if direction > 0 else 101.0
    return pd.DataFrame({"pos": [10], "strategy": ["momentum_breakout"], "direction": [direction], "stop_ref": [stop_ref]})


def test_simulate_take_profit_stop_and_tie():
    f = _flat_frame()
    f.iloc[15, f.columns.get_loc("high")] = 101.6           # target is 101.5
    won = simulate(f, _one_setup(), 1).iloc[0]
    assert won["outcome"] == "tp" and won["win"] == 1 and 1.4 < won["r"] < 1.5  # 1.5R less costs

    f = _flat_frame()
    f.iloc[15, f.columns.get_loc("low")] = 98.9             # stop is 99.0
    lost = simulate(f, _one_setup(), 1).iloc[0]
    assert lost["outcome"] == "stop" and lost["r"] < -1

    f = _flat_frame()
    f.iloc[15, f.columns.get_loc("low")] = 98.9
    f.iloc[15, f.columns.get_loc("high")] = 101.6           # both in one bar -> stop wins
    assert simulate(f, _one_setup(), 1).iloc[0]["outcome"] == "stop"

    f = _flat_frame()
    f.iloc[15, f.columns.get_loc("low")] = 98.4             # short target is 98.5
    short = simulate(f, _one_setup(-1), 1).iloc[0]
    assert short["outcome"] == "tp" and short["r"] > 1.4


def test_simulate_ignores_bars_before_entry():
    f = _flat_frame()
    f.iloc[10, f.columns.get_loc("high")] = 105.0           # the signal bar itself must not count
    assert simulate(f, _one_setup(), 1).iloc[0]["outcome"] == "time"


def test_directional_mirrors_short_features():
    X = pd.DataFrame({c: [1.0, 1.0] for c in FEATURES})
    X["d_or_high"], X["d_or_low"] = [2.0, 2.0], [5.0, 5.0]
    out = directional(X, np.array([1, -1]))
    assert out["ret1"].tolist() == [1.0, -1.0]
    assert out["adx"].tolist() == [1.0, 1.0]
    assert out["d_or_high"].tolist() == [2.0, -5.0] and out["d_or_low"].tolist() == [5.0, -2.0]


def test_live_setups_match_backtest_setups():
    """Scanning only the newest bar finds exactly what the full backtest finds there."""
    f = compute_features(resample(make_bars(), 5), 5)
    every = find_setups(f, 5)
    for pos in every["pos"].unique()[-20:]:
        live = find_setups(f.iloc[: pos + 1], 5, last_only=True)
        expected = every[every["pos"] == pos]
        assert sorted(zip(live["strategy"], live["direction"])) == sorted(zip(expected["strategy"], expected["direction"]))


def _signal(**over) -> dict:
    base = {"id": 1, "symbol": "TEST", "direction": 1, "status": "active", "entry": 100.0, "stop": 99.0,
            "tp1": 101.5, "tp2": 102.5, "risk": 1.0, "cost_r": 0.05,
            "expires_at": "2025-03-03T16:00:00+00:00"}
    return {**base, **over}


def _bars(rows):
    idx = pd.date_range("2025-03-03 10:01", periods=len(rows), freq="1min", tz=ET)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close", "volume"], index=idx).astype(float)


@pytest.fixture
def engine(tmp_path):
    db.connect(tmp_path / "test.db")
    eng = Engine()
    eng.models = {}
    return eng


def test_lifecycle_stop_target_and_runner(engine):
    now = pd.Timestamp("2025-03-03 10:05", tz=ET)
    stopped = engine.advance(_signal(), _bars([[100, 100.2, 98.9, 99, 1]]), now)
    assert stopped["status"] == "stopped" and stopped["result_r"] == -1.05

    hit = engine.advance(_signal(), _bars([[100, 101.6, 99.9, 101.4, 1]]), now)
    assert hit["status"] == "tp1_hit" and hit["result_r"] == 1.45 and "closed_at" not in hit

    runner = engine.advance(_signal(status="tp1_hit"), _bars([[101, 102.6, 100.9, 102.4, 1]]), now)
    assert runner["status"] == "tp2_hit"

    back_to_entry = engine.advance(_signal(status="tp1_hit"), _bars([[101, 101.2, 99.9, 100, 1]]), now)
    assert back_to_entry["status"] == "tp1_closed" and "result_r" not in back_to_entry


def test_lifecycle_expiry_books_the_open_result(engine):
    bars = _bars([[100, 100.6, 99.8, 100.5, 1]])
    engine.bars["TEST"] = bars
    late = pd.Timestamp("2025-03-03 12:30", tz=ET)
    out = engine.advance(_signal(expires_at="2025-03-03T17:00:00+00:00"), bars, late)  # 12:00 ET
    assert out["status"] == "expired" and out["result_r"] == 0.45
