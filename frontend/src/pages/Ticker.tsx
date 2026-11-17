import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api, type Signal } from "../api";
import Chart from "../components/Chart";
import SignalCard from "../components/SignalCard";
import { useApp } from "../live";

const TIMEFRAMES = [1, 3, 5, 15];

export function TimeframeTabs({ value, onChange }: { value: number; onChange: (tf: number) => void }) {
  return (
    <div className="tabs" role="group" aria-label="Chart timeframe">
      {TIMEFRAMES.map((tf) => (
        <button key={tf} className={tf === value ? "on" : ""} aria-pressed={tf === value} onClick={() => onChange(tf)}>
          {tf}m
        </button>
      ))}
    </div>
  );
}

export default function Ticker() {
  const symbol = (useParams().symbol || "").toUpperCase();
  const { version } = useApp();
  const [tf, setTf] = useState(5);
  const [signals, setSignals] = useState<Signal[] | null>(null);

  useEffect(() => {
    api<Signal[]>(`/signals?symbol=${symbol}&limit=50`).then(setSignals).catch(() => setSignals([]));
  }, [symbol, version]);

  return (
    <>
      <div className="title-row">
        <h1>{symbol}</h1>
        <TimeframeTabs value={tf} onChange={setTf} />
      </div>
      <Chart symbol={symbol} tf={tf} />
      <section>
        <h2>Signals for {symbol}</h2>
        {!signals ? (
          <p className="muted">Loading…</p>
        ) : signals.length === 0 ? (
          <p className="muted">No signals for {symbol} yet. It must be on your watchlist with alerts on.</p>
        ) : (
          <div className="grid">{signals.map((s) => <SignalCard key={s.id} signal={s} />)}</div>
        )}
      </section>
    </>
  );
}
