import {
  CandlestickSeries,
  createChart,
  createSeriesMarkers,
  LineStyle,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from "lightweight-charts";
import { useEffect, useRef, useState } from "react";
import { api, isOpen, type Signal } from "../api";
import { useApp } from "../live";

interface BarsResponse {
  bars: { time: number; open: number; high: number; low: number; close: number }[];
  signals: (Signal & { time: number })[];
}

const UP = "#2fd08a";
const DOWN = "#f0616d";

/** Candlestick chart with signal markers. `focus` draws that signal's entry, stop and targets. */
export default function Chart({ symbol, tf, focus }: { symbol: string; tf: number; focus?: Signal }) {
  const holder = useRef<HTMLDivElement>(null);
  const { version } = useApp();
  const [state, setState] = useState<"loading" | "empty" | "ready" | "error">("loading");

  useEffect(() => {
    if (!holder.current) return;
    let cancelled = false;
    const chart = createChart(holder.current, {
      autoSize: true,
      layout: { background: { color: "transparent" }, textColor: "#8b97a8", fontFamily: "inherit" },
      grid: { vertLines: { color: "#1a212c" }, horzLines: { color: "#1a212c" } },
      timeScale: { timeVisible: true, secondsVisible: false, borderColor: "#232c39" },
      rightPriceScale: { borderColor: "#232c39" },
    });
    const series = chart.addSeries(CandlestickSeries, {
      upColor: UP, downColor: DOWN, wickUpColor: UP, wickDownColor: DOWN, borderVisible: false,
    });

    api<BarsResponse>(`/bars/${symbol}?tf=${tf}`)
      .then((data) => {
        if (cancelled) return;
        if (!data.bars.length) return setState("empty");
        series.setData(data.bars.map((b) => ({ ...b, time: b.time as UTCTimestamp })));
        const first = data.bars[0].time;
        const markers: SeriesMarker<Time>[] = data.signals
          .filter((s) => s.timeframe === tf && s.time >= first)
          .sort((a, b) => a.time - b.time)
          .map((s) => ({
            // Snap to the bar that was closing when the signal fired
            time: (Math.floor((s.time - 1) / (tf * 60)) * tf * 60) as UTCTimestamp,
            position: s.direction > 0 ? "belowBar" : "aboveBar",
            shape: s.direction > 0 ? "arrowUp" : "arrowDown",
            color: s.direction > 0 ? UP : DOWN,
            text: s.side,
          }));
        createSeriesMarkers(series, markers);
        const shown = focus ?? data.signals.find((s) => isOpen(s) && s.timeframe === tf);
        if (shown) {
          const line = (p: number, title: string, color: string) =>
            series.createPriceLine({ price: p, title, color, lineStyle: LineStyle.Dashed, lineWidth: 1 });
          line(shown.entry, "Entry", "#8b97a8");
          line(shown.stop, "Stop", DOWN);
          line(shown.tp1, "TP1", UP);
          line(shown.tp2, "TP2", UP);
        }
        chart.timeScale().fitContent();
        setState("ready");
      })
      .catch(() => !cancelled && setState("error"));

    return () => {
      cancelled = true;
      chart.remove();
    };
  }, [symbol, tf, focus?.id, focus?.status, version]);

  return (
    <div className="chart-wrap">
      <div className="chart" ref={holder} />
      {state === "loading" && <p className="chart-note">Loading chart…</p>}
      {state === "empty" && <p className="chart-note">No price data for {symbol} yet.</p>}
      {state === "error" && <p className="chart-note">The chart could not be loaded. Reload the page to try again.</p>}
    </div>
  );
}
