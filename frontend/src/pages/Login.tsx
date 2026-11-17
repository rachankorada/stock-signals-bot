import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { post, setToken } from "../api";
import { useApp } from "../live";

export default function Login() {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const { refreshUser } = useApp();
  const navigate = useNavigate();

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const { token } = await post<{ token: string }>(`/auth/${mode}`, { email, password });
      setToken(token);
      await refreshUser();
      navigate(mode === "register" ? "/watchlist" : "/");
    } catch (err) {
      setError((err as Error).message);
    }
    setBusy(false);
  }

  return (
    <div className="center">
      <form className="card auth" onSubmit={submit}>
        <h1>Stock Analyzer</h1>
        <p className="muted">Intraday buy and sell signals for the tickers you choose.</p>
        <label>
          Email
          <input type="email" autoComplete="email" required value={email} onChange={(e) => setEmail(e.target.value)} />
        </label>
        <label>
          Password
          <input
            type="password"
            autoComplete={mode === "login" ? "current-password" : "new-password"}
            required
            minLength={8}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
          {mode === "register" && <small>At least 8 characters.</small>}
        </label>
        {error && <p className="error" role="alert">{error}</p>}
        <button className="primary" disabled={busy}>
          {mode === "login" ? "Sign in" : "Create account"}
        </button>
        <button type="button" className="link" onClick={() => { setMode(mode === "login" ? "register" : "login"); setError(""); }}>
          {mode === "login" ? "New here? Create an account" : "Already have an account? Sign in"}
        </button>
      </form>
    </div>
  );
}
