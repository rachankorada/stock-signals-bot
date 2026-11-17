import { useEffect, useState, type FormEvent } from "react";
import { Link } from "react-router-dom";
import { api, post } from "../api";

interface Item {
  symbol: string;
  enabled: number;
}

export default function Watchlist() {
  const [items, setItems] = useState<Item[] | null>(null);
  const [max, setMax] = useState(25);
  const [symbol, setSymbol] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const load = () =>
    api<{ items: Item[]; max: number }>("/watchlist")
      .then((d) => {
        setItems(d.items);
        setMax(d.max);
      })
      .catch((e) => setError(e.message));

  useEffect(() => {
    load();
  }, []);

  async function add(e: FormEvent) {
    e.preventDefault();
    // Accept several tickers at once: "AAPL, NVDA TSLA"
    const symbols = symbol.toUpperCase().split(/[\s,]+/).filter(Boolean);
    if (!symbols.length) return;
    setBusy(true);
    setError("");
    const failed: string[] = [];
    for (const s of symbols) {
      try {
        await post("/watchlist", { symbol: s });
      } catch (err) {
        failed.push((err as Error).message);
      }
    }
    setSymbol("");
    setError(failed.join(" "));
    await load();
    setBusy(false);
  }

  async function toggle(item: Item) {
    setItems((list) => list!.map((i) => (i.symbol === item.symbol ? { ...i, enabled: i.enabled ? 0 : 1 } : i)));
    await api(`/watchlist/${item.symbol}`, { method: "PATCH", body: JSON.stringify({ enabled: !item.enabled }) }).catch(load);
  }

  async function remove(item: Item) {
    setItems((list) => list!.filter((i) => i.symbol !== item.symbol));
    await api(`/watchlist/${item.symbol}`, { method: "DELETE" }).catch(load);
  }

  if (!items) return <p className="muted">{error || "Loading watchlist…"}</p>;

  return (
    <>
      <h1>Watchlist</h1>
      <p className="muted">
        You only get signals for the tickers listed here. {items.length} of {max} used.
      </p>
      <form className="row" onSubmit={add}>
        <input
          aria-label="Ticker symbols to add"
          placeholder="AAPL, NVDA, TSLA"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
          autoCapitalize="characters"
          autoComplete="off"
        />
        <button className="primary" disabled={busy || !symbol.trim()}>
          {busy ? "Adding…" : "Add"}
        </button>
      </form>
      {error && <p className="error" role="alert">{error}</p>}
      {items.length === 0 ? (
        <div className="card empty">
          <p>Your watchlist is empty. Add US stock or ETF tickers above, separated by commas.</p>
        </div>
      ) : (
        <ul className="list">
          {items.map((item) => (
            <li key={item.symbol} className={item.enabled ? "" : "off"}>
              <Link to={`/ticker/${item.symbol}`} className="ticker">{item.symbol}</Link>
              <label className="switch">
                <input type="checkbox" checked={!!item.enabled} onChange={() => toggle(item)} />
                <span>{item.enabled ? "Alerts on" : "Alerts paused"}</span>
              </label>
              <button className="ghost" onClick={() => remove(item)} aria-label={`Remove ${item.symbol}`}>
                Remove
              </button>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}
