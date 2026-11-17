import { useEffect, useState } from "react";
import { api, enablePush, post, pushEnabledHere, type UserSettings } from "../api";
import { useApp } from "../live";

const TIMEFRAMES = [1, 3, 5, 15];

export default function SettingsPage() {
  const { user, refreshUser, signOut } = useApp();
  const [settings, setSettings] = useState<UserSettings>(user!.settings);
  const [saved, setSaved] = useState(false);
  const [pushOn, setPushOn] = useState(false);
  const [pushMsg, setPushMsg] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    pushEnabledHere().then(setPushOn);
  }, []);

  async function save(next: UserSettings) {
    setSettings(next);
    setSaved(false);
    setError("");
    try {
      await api("/settings", { method: "PUT", body: JSON.stringify(next) });
      await refreshUser();
      setSaved(true);
    } catch (e) {
      setError((e as Error).message);
    }
  }

  async function turnOnPush() {
    setPushMsg("");
    try {
      await enablePush();
      setPushOn(true);
      setPushMsg("Notifications are on for this device.");
    } catch (e) {
      setPushMsg((e as Error).message);
    }
  }

  async function testPush() {
    const { sent } = await post<{ sent: number }>("/push/test");
    setPushMsg(sent ? "Test notification sent." : "No device is registered. Turn notifications on first.");
  }

  const toggleTf = (tf: number) =>
    save({
      ...settings,
      timeframes: settings.timeframes.includes(tf)
        ? settings.timeframes.filter((t) => t !== tf)
        : [...settings.timeframes, tf].sort((a, b) => a - b),
    });

  return (
    <>
      <h1>Settings</h1>
      <div className="card form">
        <h2>Which signals you get</h2>
        <label>
          Minimum confidence: {Math.max(settings.min_confidence, 50)}%
          <input
            type="range" min={50} max={90} step={5} value={Math.max(settings.min_confidence, 50)}
            onChange={(e) => setSettings({ ...settings, min_confidence: Number(e.target.value) })}
            onPointerUp={() => save(settings)} onKeyUp={() => save(settings)}
          />
          <small>Only signals at 50% confidence or higher are ever posted. Raise this to get fewer, higher-confidence signals.</small>
        </label>
        <fieldset>
          <legend>Timeframes</legend>
          {TIMEFRAMES.map((tf) => (
            <label key={tf} className="check">
              <input type="checkbox" checked={settings.timeframes.includes(tf)} onChange={() => toggleTf(tf)} />
              {tf} minute
            </label>
          ))}
        </fieldset>
        <label className="check">
          <input type="checkbox" checked={settings.long_only} onChange={(e) => save({ ...settings, long_only: e.target.checked })} />
          BUY signals only (hide short SELL signals)
        </label>
        {saved && <p className="ok" role="status">Saved.</p>}
        {error && <p className="error" role="alert">{error}</p>}
      </div>

      <div className="card form">
        <h2>Notifications</h2>
        <label className="check">
          <input type="checkbox" checked={settings.sound} onChange={(e) => save({ ...settings, sound: e.target.checked })} />
          Play a sound for alerts while the site is open
        </label>
        <label className="check">
          <input type="checkbox" checked={settings.push} onChange={(e) => save({ ...settings, push: e.target.checked })} />
          Send notifications to my devices
        </label>
        <div className="row">
          <button className="primary" onClick={turnOnPush} disabled={pushOn}>
            {pushOn ? "On for this device" : "Turn on for this device"}
          </button>
          <button className="ghost" onClick={testPush}>Send a test</button>
        </div>
        {pushMsg && <p className="muted" role="status">{pushMsg}</p>}
        <small>
          Each phone or computer must be turned on separately. On iPhone, open this site in Safari, tap Share, choose
          Add to Home Screen, then open it from the Home Screen and turn notifications on.
        </small>
      </div>

      <div className="card form">
        <h2>Account</h2>
        <p className="muted">Signed in as {user!.email}</p>
        <button className="ghost" onClick={signOut}>Sign out</button>
      </div>
    </>
  );
}
