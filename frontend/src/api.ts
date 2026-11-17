const TOKEN_KEY = "sa_token";

export const getToken = () => localStorage.getItem(TOKEN_KEY);
export const setToken = (token: string | null) =>
  token ? localStorage.setItem(TOKEN_KEY, token) : localStorage.removeItem(TOKEN_KEY);

export async function api<T = any>(path: string, options: RequestInit = {}): Promise<T> {
  const token = getToken();
  const res = await fetch("/api" + path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(token ? { Authorization: "Bearer " + token } : {}),
      ...(options.headers || {}),
    },
  });
  if (res.status === 401 && token && !path.startsWith("/auth")) {
    setToken(null);
    window.location.href = "/login";
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || "Something went wrong. Try again.");
  return data as T;
}

export const post = <T = any>(path: string, body?: unknown) =>
  api<T>(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export interface Signal {
  id: number;
  symbol: string;
  timeframe: number;
  direction: number;
  side: "BUY" | "SELL";
  strategy: string;
  strategy_label: string;
  agreeing: string[];
  entry: number;
  entry_max: number;
  stop: number;
  tp1: number;
  tp2: number;
  risk: number;
  cost_r: number;
  confidence: number;
  reasons: string[];
  history: { trades?: number; win_rate?: number; expectancy_r?: number };
  validated: number;
  status: "active" | "tp1_hit" | "tp2_hit" | "tp1_closed" | "stopped" | "expired";
  created_at: string;
  expires_at: string;
  closed_at: string | null;
  exit_price: number | null;
  result_r: number | null;
}

export interface UserSettings {
  min_confidence: number;
  timeframes: number[];
  long_only: boolean;
  push: boolean;
  sound: boolean;
}

export interface User {
  id: number;
  email: string;
  settings: UserSettings;
}

export const STATUS_LABEL: Record<Signal["status"], string> = {
  active: "Active",
  tp1_hit: "Target 1 hit",
  tp2_hit: "Target 2 hit",
  tp1_closed: "Closed after target 1",
  stopped: "Stopped out",
  expired: "Expired",
};

export const isOpen = (s: Signal) => s.status === "active" || s.status === "tp1_hit";

const NY = "America/New_York";
export const timeET = (iso: string) =>
  new Date(iso).toLocaleTimeString("en-US", { timeZone: NY, hour: "numeric", minute: "2-digit" });
export const dateET = (iso: string) =>
  new Date(iso).toLocaleDateString("en-US", { timeZone: NY, month: "short", day: "numeric" });
export const price = (n: number) => n.toFixed(2);
export const pct = (n: number) => (n * 100).toFixed(0) + "%";
export const signedR = (n: number) => (n >= 0 ? "+" : "") + n.toFixed(2) + "R";

// ---- device notifications ----

function keyBytes(base64: string) {
  const padded = (base64 + "=".repeat((4 - (base64.length % 4)) % 4)).replace(/-/g, "+").replace(/_/g, "/");
  return Uint8Array.from(atob(padded), (c) => c.charCodeAt(0));
}

export const pushSupported = () =>
  "serviceWorker" in navigator && "PushManager" in window && "Notification" in window;

export async function enablePush() {
  if (!pushSupported()) {
    throw new Error("This browser cannot receive push notifications. On iPhone, add this site to your Home Screen first.");
  }
  const permission = await Notification.requestPermission();
  if (permission !== "granted") {
    throw new Error("Notifications are blocked. Allow them for this site in your browser settings.");
  }
  const registration = await navigator.serviceWorker.ready;
  const { key } = await api<{ key: string }>("/push/key");
  const sub = await registration.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: keyBytes(key) });
  await post("/push/subscribe", sub.toJSON());
}

export async function pushEnabledHere() {
  if (!pushSupported() || Notification.permission !== "granted") return false;
  const registration = await navigator.serviceWorker.getRegistration();
  return !!(registration && (await registration.pushManager.getSubscription()));
}
