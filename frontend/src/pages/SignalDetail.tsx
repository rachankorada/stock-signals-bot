import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, type Signal } from "../api";
import Chart from "../components/Chart";
import SignalCard from "../components/SignalCard";
import { useApp } from "../live";

export default function SignalDetail() {
  const { id } = useParams();
  const { version } = useApp();
  const [signal, setSignal] = useState<Signal | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api<Signal>(`/signals/${id}`).then(setSignal).catch((e) => setError(e.message));
  }, [id, version]);

  if (error) return <p className="error">{error}</p>;
  if (!signal) return <p className="muted">Loading signal…</p>;

  return (
    <>
      <div className="title-row">
        <h1>
          {signal.side} {signal.symbol}
        </h1>
        <Link to={`/ticker/${signal.symbol}`} className="button ghost">All {signal.symbol} signals</Link>
      </div>
      <SignalCard signal={signal} full />
      <Chart symbol={signal.symbol} tf={signal.timeframe} focus={signal} />
    </>
  );
}
