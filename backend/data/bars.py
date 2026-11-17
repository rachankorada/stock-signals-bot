"""Build 3, 5 and 15-minute bars from 1-minute bars."""
import pandas as pd

SESSION_OPEN_MIN = 570   # 09:30 ET
SESSION_CLOSE_MIN = 960  # 16:00 ET
SESSION_LEN = SESSION_CLOSE_MIN - SESSION_OPEN_MIN


def session_minutes(index: pd.DatetimeIndex):
    """Minutes since the 09:30 open for each bar start."""
    return index.hour * 60 + index.minute - SESSION_OPEN_MIN


def resample(df1: pd.DataFrame, tf: int) -> pd.DataFrame:
    """Aggregate 1-minute bars into tf-minute bars anchored at 09:30.

    The same function serves history and live data so both always match.
    The caller must pass only complete buckets.
    """
    if tf == 1 or df1.empty:
        return df1
    mins = session_minutes(df1.index)
    start = df1.index.normalize() + pd.to_timedelta((mins // tf) * tf + SESSION_OPEN_MIN, unit="m")
    out = df1.groupby(start).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
    )
    out.index.name = df1.index.name
    return out
