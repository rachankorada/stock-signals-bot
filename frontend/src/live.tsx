import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { Link } from "react-router-dom";
import { api, getToken, pct, price, setToken, STATUS_LABEL, type Signal, type User } from "./api";

interface Toast {
  key: number;
  signal: Signal;
  isNew: boolean;
}

interface AppState {
  user: User | null;
  loading: boolean;
  refreshUser: () => Promise<void>;
  signOut: () => void;
  /** Increases whenever a live event arrives, so pages know to reload. */
  version: number;
  connected: boolean;
}

const Ctx = createContext<AppState>(null as unknown as AppState);
export const useApp = () => useContext(Ctx);

function beep(up: boolean) {
  try {
    const audio = new AudioContext();
    const osc = audio.createOscillator();
    const gain = audio.createGain();
    osc.frequency.value = up ? 880 : 520;
    gain.gain.setValueAtTime(0.15, audio.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audio.currentTime + 0.4);
    osc.connect(gain).connect(audio.destination);
    osc.start();
    osc.stop(audio.currentTime + 0.4);
  } catch {
    // Browsers block sound until the first click on the page
  }
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [version, setVersion] = useState(0);
  const [connected, setConnected] = useState(false);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const soundRef = useRef(true);

  const refreshUser = useCallback(async () => {
    if (!getToken()) {
      setUser(null);
      setLoading(false);
      return;
    }
    try {
      const me = await api<User>("/me");
      soundRef.current = me.settings.sound;
      setUser(me);
    } catch {
      setUser(null);
    }
    setLoading(false);
  }, []);

  const signOut = useCallback(() => {
    setToken(null);
    setUser(null);
  }, []);

  useEffect(() => {
    refreshUser();
  }, [refreshUser]);

  // Live alert socket, reconnecting with a growing delay
  const userId = user?.id;
  useEffect(() => {
    if (!userId) return;
    let socket: WebSocket | null = null;
    let closed = false;
    let retry = 1000;
    let timer = 0;
    let ping = 0;
    const open = () => {
      const scheme = location.protocol === "https:" ? "wss" : "ws";
      socket = new WebSocket(`${scheme}://${location.host}/ws?token=${getToken()}`);
      socket.onopen = () => {
        retry = 1000;
        setConnected(true);
        ping = window.setInterval(() => socket?.readyState === 1 && socket.send("ping"), 25000);
      };
      socket.onmessage = (msg) => {
        const event = JSON.parse(msg.data) as { type: "signal" | "update"; signal: Signal };
        setVersion((v) => v + 1);
        const toast = { key: Date.now() + Math.random(), signal: event.signal, isNew: event.type === "signal" };
        setToasts((list) => [toast, ...list].slice(0, 4));
        window.setTimeout(() => setToasts((list) => list.filter((t) => t.key !== toast.key)), 15000);
        if (soundRef.current) beep(event.signal.direction > 0);
      };
      socket.onclose = () => {
        setConnected(false);
        window.clearInterval(ping);
        if (!closed) timer = window.setTimeout(open, retry);
        retry = Math.min(retry * 2, 30000);
      };
    };
    open();
    return () => {
      closed = true;
      window.clearTimeout(timer);
      window.clearInterval(ping);
      socket?.close();
    };
  }, [userId]);

  return (
    <Ctx.Provider value={{ user, loading, refreshUser, signOut, version, connected }}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        {toasts.map((t) => (
          <Link
            key={t.key}
            to={`/signal/${t.signal.id}`}
            className={`toast ${t.signal.direction > 0 ? "buy" : "sell"}`}
            onClick={() => setToasts((list) => list.filter((x) => x.key !== t.key))}
          >
            <strong>
              {t.signal.side} {t.signal.symbol}
              {t.isNew ? ` @ ${price(t.signal.entry)}` : `: ${STATUS_LABEL[t.signal.status]}`}
            </strong>
            <span>
              {t.isNew
                ? `SL ${price(t.signal.stop)} · TP ${price(t.signal.tp1)} · ${pct(t.signal.confidence)} · ${t.signal.timeframe}m`
                : `${t.signal.timeframe}m signal from ${price(t.signal.entry)}`}
            </span>
          </Link>
        ))}
      </div>
    </Ctx.Provider>
  );
}
