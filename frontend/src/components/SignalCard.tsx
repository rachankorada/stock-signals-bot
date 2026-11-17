import { Link } from "react-router-dom";
import { dateET, isOpen, pct, price, signedR, STATUS_LABEL, timeET, type Signal } from "../api";

function statusClass(s: Signal) {
  if (s.status === "stopped") return "bad";
  if (s.status === "expired") return (s.result_r ?? 0) >= 0 ? "good" : "bad";
  if (s.status === "active") return "live";
  return "good";
}

export function Level({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="level">
      <span>{label}</span>
      <strong className={tone}>{value}</strong>
    </div>
  );
}

/** One signal with its trade levels. `full` adds the reasons and track record. */
export default function SignalCard({ signal: s, full = false }: { signal: Signal; full?: boolean }) {
  const buy = s.direction > 0;
  const body = (
    <>
      <div className="card-head">
        <span className={`side ${buy ? "buy" : "sell"}`}>{s.side}</span>
        <span className="ticker">{s.symbol}</span>
        <span className="chip">{s.timeframe}m</span>
        {!s.validated && <span className="chip warn" title="This setup has not passed backtest validation">Unvalidated</span>}
        <span className={`status ${statusClass(s)}`}>
          {STATUS_LABEL[s.status]}
          {s.result_r != null && ` ${signedR(s.result_r)}`}
        </span>
      </div>
      <div className="levels">
        <Level label="Entry" value={price(s.entry)} />
        <Level label="Stop loss" value={price(s.stop)} tone="neg" />
        <Level label="Target 1" value={price(s.tp1)} tone="pos" />
        <Level label="Target 2" value={price(s.tp2)} tone="pos" />
        <Level label="Confidence" value={pct(s.confidence)} />
      </div>
      <div className="card-foot">
        <span>{s.strategy_label}</span>
        <span>
          {dateET(s.created_at)}, {timeET(s.created_at)} ET
          {isOpen(s) && ` · expires ${timeET(s.expires_at)}`}
        </span>
      </div>
    </>
  );

  if (!full) {
    return (
      <Link to={`/signal/${s.id}`} className={`card signal ${buy ? "buy" : "sell"}`}>
        {body}
      </Link>
    );
  }

  const risk = Math.abs(s.entry - s.stop);
  return (
    <div className={`card signal ${buy ? "buy" : "sell"}`}>
      {body}
      <dl className="facts">
        <div>
          <dt>Enter no {buy ? "higher" : "lower"} than</dt>
          <dd>{price(s.entry_max)}</dd>
        </div>
        <div>
          <dt>Risk per share</dt>
          <dd>${price(risk)}</dd>
        </div>
        <div>
          <dt>Risk : reward</dt>
          <dd>
            1 : {(Math.abs(s.tp1 - s.entry) / risk).toFixed(1)} and 1 : {(Math.abs(s.tp2 - s.entry) / risk).toFixed(1)}
          </dd>
        </div>
        <div>
          <dt>Estimated trading cost</dt>
          <dd>{s.cost_r.toFixed(2)}R</dd>
        </div>
        {s.history.trades ? (
          <div>
            <dt>This setup in backtests</dt>
            <dd>
              {pct(s.history.win_rate ?? 0)} hit target 1 over {s.history.trades.toLocaleString()} trades
            </dd>
          </div>
        ) : null}
        {s.exit_price != null && (
          <div>
            <dt>Exit price</dt>
            <dd>{price(s.exit_price)}</dd>
          </div>
        )}
      </dl>
      <h3>Why this signal</h3>
      <ul className="reasons">
        {s.reasons.map((r) => (
          <li key={r}>{r}</li>
        ))}
        {s.agreeing.map((a) => (
          <li key={a}>Also confirmed by: {a}</li>
        ))}
      </ul>
      <p className="hint">
        Confidence is the model's estimate that target 1 is reached before the stop loss. The stop moves to the entry
        price once target 1 is hit.
      </p>
    </div>
  );
}
