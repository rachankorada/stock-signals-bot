import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, isOpen, signedR, timeET, type Signal } from "../api";
import SignalCard from "../components/SignalCard";
import { useApp } from "../live";

interface Market {
  is_open: boolean;
  session_open: string | null;
  engine_running: boolean;
  last_scan: string | null;
  error: string | null;
  symbols: number;
  models: { timeframe: number; validated: boolean }[];
  allow_unvalidated: boolean;
  min_post_confidence: number;
}

function MarketBanner({ market }: { market: Market }) {
  const live = market.models.filter((m) => m.validated).map((m) => `${m.timeframe}m`);
  let text: string;
  if (market.error) text = "The live data connection has a problem. Signals may be delayed.";
  else if (!market.engine_running) text = "The signal engine is not running.";
  else if (market.is_open)
    text = `Market open. Scanning ${market.symbols} ${market.symbols === 1 ? "ticker" : "tickers"}${
      market.last_scan ? `, last check ${timeET(market.last_scan)} ET` : ""
    }.`;
  else text = "Market closed. Signals resume at 9:30 AM ET on the next trading day.";
  return (
    <div className={`banner ${market.is_open && !market.error ? "open" : ""}`}>
      <span>{text}</span>
      <span className="muted">
        {live.length ? `Validated timeframes: ${live.join(", ")}` : "No timeframe has passed validation yet"}
        {market.allow_unvalidated && " · unvalidated signals are on"}
        {` · only ${Math.round(market.min_post_confidence * 100)}%+ confidence is posted`}
      </span>
    </div>
  );
}

export default function Dashboard() {
  const { version } = useApp();
  const [signals, setSignals] = useState<Signal[] | null>(null);
  const [market, setMarket] = useState<Market | null>(null);
  const [watchCount, setWatchCount] = useState<number | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      Promise.all([api<Signal[]>("/signals?limit=60"), api<Market>("/market"), api<{ items: unknown[] }>("/watchlist")])
        .then(([s, m, w]) => {
          if (cancelled) return;
          setSignals(s);
          setMarket(m);
          setWatchCount(w.items.length);
          setError("");
        })
        .catch((e) => !cancelled && setError(e.message));
    };
    load();
    const timer = window.setInterval(load, 20000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [version]);

  if (error && !signals) return <p className="error">{error}</p>;
  if (!signals || !market) return <p className="muted">Loading signals…</p>;

  const open = signals.filter(isOpen);
  const today = new Date().toDateString();
  const closed = signals.filter((s) => !isOpen(s));
  const closedToday = closed.filter((s) => new Date(s.created_at).toDateString() === today);
  const todayR = closedToday.reduce((sum, s) => sum + (s.result_r ?? 0), 0);

  return (
    <>
      <MarketBanner market={market} />
      {watchCount === 0 && (
        <div className="card empty">
          <h2>Add tickers to start</h2>
          <p>Signals are only generated for the tickers on your watchlist.</p>
          <Link to="/watchlist" className="primary button">Build your watchlist</Link>
        </div>
      )}
      <section>
        <h2>Open signals {open.length > 0 && <span className="count">{open.length}</span>}</h2>
        {open.length === 0 ? (
          <p className="muted">
            {watchCount === 0
              ? "Nothing yet."
              : market.is_open
              ? "No open signals right now. New ones appear here the moment they are detected."
              : "No open signals. The market is closed."}
          </p>
        ) : (
          <div className="grid">{open.map((s) => <SignalCard key={s.id} signal={s} />)}</div>
        )}
      </section>
      <section>
        <h2>
          Recent results
          {closedToday.length > 0 && (
            <span className={`count ${todayR >= 0 ? "pos" : "neg"}`}>
              today {signedR(todayR)} over {closedToday.length}
            </span>
          )}
        </h2>
        {closed.length === 0 ? (
          <p className="muted">Finished signals and their outcomes will be listed here.</p>
        ) : (
          <div className="grid">{closed.slice(0, 20).map((s) => <SignalCard key={s.id} signal={s} />)}</div>
        )}
      </section>
    </>
  );
}
