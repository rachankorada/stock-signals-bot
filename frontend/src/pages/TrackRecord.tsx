import { useEffect, useState } from "react";
import { api, pct, signedR } from "../api";
import { useApp } from "../live";

interface Group {
  key: string;
  trades: number;
  win_rate: number;
  expectancy_r: number;
  total_r: number;
}

interface Stats {
  trades: number;
  win_rate?: number;
  expectancy_r?: number;
  total_r?: number;
  profit_factor?: number | null;
  by_strategy?: Group[];
  by_timeframe?: Group[];
  by_confidence?: Group[];
  by_side?: Group[];
  equity?: { time: string; cum_r: number }[];
}

interface Backtest {
  trades: number;
  win_rate?: number;
  expectancy_r?: number;
  profit_factor?: number | null;
}

interface Model {
  timeframe: number;
  validated: boolean;
  p_min: number;
  trained_through: string;
  backtest: {
    all: Backtest;
    one_per_ticker: Backtest;
    final_six_months: Backtest;
    per_day: number;
    by_year: (Backtest & { year: number })[];
    start: string;
    end: string;
  };
  geometry: { tp1_r: number; tp2_r: number };
  raw: Backtest;
  strategies: Record<string, Backtest & { enabled: boolean }>;
  calibration: { bucket: string; n: number; predicted: number; actual: number }[];
}

function Tile({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="tile">
      <span>{label}</span>
      <strong className={tone}>{value}</strong>
    </div>
  );
}

function GroupTable({ title, head, rows }: { title: string; head: string; rows: Group[] }) {
  if (!rows.length) return null;
  return (
    <div className="card">
      <h3>{title}</h3>
      <table>
        <thead>
          <tr><th>{head}</th><th>Signals</th><th>Win rate</th><th>Avg result</th><th>Total</th></tr>
        </thead>
        <tbody>
          {rows.map((g) => (
            <tr key={g.key}>
              <td>{g.key}</td>
              <td>{g.trades}</td>
              <td>{pct(g.win_rate)}</td>
              <td className={g.expectancy_r >= 0 ? "pos" : "neg"}>{signedR(g.expectancy_r)}</td>
              <td className={g.total_r >= 0 ? "pos" : "neg"}>{signedR(g.total_r)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Cumulative result line, drawn as a simple SVG path. */
function Equity({ points }: { points: { cum_r: number }[] }) {
  if (points.length < 2) return null;
  const values = [0, ...points.map((p) => p.cum_r)];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const x = (i: number) => (i / (values.length - 1)) * 600;
  const y = (v: number) => 150 - ((v - lo) / span) * 140 - 5;
  const path = values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
  const up = values[values.length - 1] >= 0;
  return (
    <div className="card">
      <h3>Cumulative result (R)</h3>
      <svg viewBox="0 0 600 150" className="equity" role="img" aria-label={`Cumulative result ${signedR(values[values.length - 1])}`}>
        <line x1="0" x2="600" y1={y(0)} y2={y(0)} stroke="#2a3443" strokeDasharray="4 4" />
        <path d={path} fill="none" stroke={up ? "#2fd08a" : "#f0616d"} strokeWidth="2" />
      </svg>
    </div>
  );
}

function Result({ value }: { value?: number }) {
  if (value == null) return <td>–</td>;
  return <td className={value >= 0 ? "pos" : "neg"}>{signedR(value)}</td>;
}

function ModelCard({ m }: { m: Model }) {
  const b = m.backtest;
  const seen = b.one_per_ticker;
  return (
    <div className="card">
      <div className="card-head">
        <span className="ticker">{m.timeframe}-minute model</span>
        <span className={`status ${m.validated ? "good" : "bad"}`}>{m.validated ? "Validated" : "Not validated"}</span>
      </div>
      <p className="muted">
        Backtest of posted signals ({pct(m.p_min)} confidence and above), {b.start} to {b.end}, 25 tickers, after
        costs. Every signal was scored by a model trained only on earlier data.
      </p>
      {seen.trades ? (
        <div className="tiles">
          <Tile label="Signals" value={seen.trades.toLocaleString()} />
          <Tile label="Per day" value={String(b.per_day)} />
          <Tile label="Win rate" value={pct(seen.win_rate ?? 0)} />
          <Tile label="Average result" value={signedR(seen.expectancy_r ?? 0)} tone={(seen.expectancy_r ?? 0) >= 0 ? "pos" : "neg"} />
        </div>
      ) : (
        <p className="muted">No setup reached {pct(m.p_min)} confidence in the backtest.</p>
      )}
      <table>
        <thead>
          <tr><th>Period</th><th>Signals</th><th>Win rate</th><th>Avg result</th></tr>
        </thead>
        <tbody>
          {b.by_year.map((y) => (
            <tr key={y.year}>
              <td>{y.year}</td>
              <td>{y.trades}</td>
              <td>{y.trades ? pct(y.win_rate ?? 0) : "–"}</td>
              <Result value={y.trades ? y.expectancy_r : undefined} />
            </tr>
          ))}
          <tr>
            <td>Final 6 months</td>
            <td>{b.final_six_months.trades}</td>
            <td>{b.final_six_months.trades ? pct(b.final_six_months.win_rate ?? 0) : "–"}</td>
            <Result value={b.final_six_months.trades ? b.final_six_months.expectancy_r : undefined} />
          </tr>
          <tr>
            <td>Every setup, no model</td>
            <td>{m.raw.trades.toLocaleString()}</td>
            <td>{pct(m.raw.win_rate ?? 0)}</td>
            <Result value={m.raw.expectancy_r} />
          </tr>
        </tbody>
      </table>
      <details>
        <summary>By strategy</summary>
        <table>
          <thead>
            <tr><th>Strategy</th><th>Signals</th><th>Win rate</th><th>Avg result</th></tr>
          </thead>
          <tbody>
            {Object.entries(m.strategies).map(([name, s]) => (
              <tr key={name}>
                <td>{name}</td>
                <td>{s.trades}</td>
                <td>{s.trades ? pct(s.win_rate ?? 0) : "–"}</td>
                <Result value={s.trades ? s.expectancy_r : undefined} />
              </tr>
            ))}
          </tbody>
        </table>
      </details>
      <details>
        <summary>Confidence calibration</summary>
        <table>
          <thead>
            <tr><th>Confidence</th><th>Setups</th><th>Predicted</th><th>Actual win rate</th></tr>
          </thead>
          <tbody>
            {m.calibration.map((c) => (
              <tr key={c.bucket}>
                <td>{c.bucket}</td>
                <td>{c.n.toLocaleString()}</td>
                <td>{pct(c.predicted)}</td>
                <td>{pct(c.actual)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}

export default function TrackRecord() {
  const { version } = useApp();
  const [stats, setStats] = useState<Stats | null>(null);
  const [models, setModels] = useState<Model[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([api<Stats>("/stats"), api<{ models: Model[] }>("/market")])
      .then(([s, m]) => {
        setStats(s);
        setModels(m.models);
      })
      .catch((e) => setError(e.message));
  }, [version]);

  if (error) return <p className="error">{error}</p>;
  if (!stats) return <p className="muted">Loading track record…</p>;

  return (
    <>
      <h1>Track record</h1>
      <p className="muted">
        Results of every finished signal on your tickers, after estimated trading costs. 1R is the amount risked
        between entry and stop loss. A win is booked at target 1.
      </p>
      {stats.trades === 0 ? (
        <div className="card empty"><p>No finished signals yet. Results appear here as live signals close.</p></div>
      ) : (
        <>
          <div className="tiles">
            <Tile label="Signals" value={String(stats.trades)} />
            <Tile label="Win rate" value={pct(stats.win_rate!)} />
            <Tile label="Average result" value={signedR(stats.expectancy_r!)} tone={stats.expectancy_r! >= 0 ? "pos" : "neg"} />
            <Tile label="Total" value={signedR(stats.total_r!)} tone={stats.total_r! >= 0 ? "pos" : "neg"} />
            <Tile label="Profit factor" value={stats.profit_factor == null ? "–" : stats.profit_factor.toFixed(2)} />
          </div>
          <Equity points={stats.equity!} />
          <div className="grid">
            <GroupTable title="By strategy" head="Strategy" rows={stats.by_strategy!} />
            <GroupTable title="By timeframe" head="Minutes" rows={stats.by_timeframe!} />
            <GroupTable title="By confidence" head="Confidence" rows={stats.by_confidence!} />
            <GroupTable title="By side" head="Side" rows={stats.by_side!} />
          </div>
        </>
      )}
      <section>
        <h2>Model validation</h2>
        <p className="muted">
          Three-year backtest of each model. A timeframe counts as validated only if its posted signals were
          profitable after costs, including over the final six months.
        </p>
        {models.length === 0 ? (
          <p className="muted">No trained models found.</p>
        ) : (
          <div className="grid wide">{models.map((m) => <ModelCard key={m.timeframe} m={m} />)}</div>
        )}
      </section>
    </>
  );
}
