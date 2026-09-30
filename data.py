"""Market data. V0.1 ships a seeded regime-switching synthetic market so the lab runs offline
and deterministically. Swap in real bars via load_csv or an Alpaca data fetcher later."""
from __future__ import annotations
import csv
import numpy as np


def synthetic_market(n: int = 3000, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    # regimes: 0 calm-uptrend, 1 chop, 2 high-vol, 3 downtrend
    params = {0: (0.0006, 0.007), 1: (0.0000, 0.009), 2: (0.0000, 0.022), 3: (-0.0006, 0.012)}
    regime = np.zeros(n, dtype=int)
    r = 0
    for i in range(n):
        if rng.random() < 0.012:
            r = int(rng.integers(0, 4))
        regime[i] = r
    mu = np.array([params[x][0] for x in regime])
    sd = np.array([params[x][1] for x in regime])
    rets = mu + sd * rng.standard_normal(n)
    close = 100.0 * np.cumprod(1.0 + rets)
    volume = 1e6 * np.exp(0.3 * rng.standard_normal(n) + 0.8 * (sd / 0.01 - 1.0) * 0.3)
    return {"close": close, "volume": volume, "regime": regime}


def load_csv(path: str) -> dict:
    close, volume = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            close.append(float(row["close"]))
            volume.append(float(row["volume"]))
    return {"close": np.array(close), "volume": np.array(volume)}


def slice_bars(bars: dict, a: int, b: int) -> dict:
    return {k: v[a:b] for k, v in bars.items()}


# ---------------------------------------------------------------------------- real data (Alpaca)
import datetime as _dt
import json
import os
import urllib.request

DATA_HOST = "https://data.alpaca.markets"


class DataQualityError(RuntimeError):
    pass


def load_bars_csv(path: str) -> dict:
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append((row["date"], float(row["open"]), float(row["high"]), float(row["low"]),
                         float(row["close"]), float(row["volume"])))
    rows.sort()
    return {"dates": np.array([r[0] for r in rows]), "close": np.array([r[4] for r in rows]),
            "volume": np.array([r[5] for r in rows]), "open": np.array([r[1] for r in rows]),
            "high": np.array([r[2] for r in rows]), "low": np.array([r[3] for r in rows])}


def check_quality(bars: dict, today: str | None = None, max_stale_days: int = 5, max_gap_days: int = 7,
                  max_abs_ret: float = 0.5, min_bars: int = 300) -> tuple:
    """Return (fatal_issues, warnings). Any fatal issue should trigger the governor's data-quality HALT."""
    fatal, warn = [], []
    dates, close, vol = bars["dates"], bars["close"], bars["volume"]
    if len(close) < min_bars:
        fatal.append(f"too_few_bars:{len(close)}")
        return fatal, warn
    if not np.isfinite(close).all() or (close <= 0).any():
        fatal.append("bad_prices")
        return fatal, warn
    d = [_dt.date.fromisoformat(x) for x in dates]
    if any(b <= a for a, b in zip(d, d[1:])):
        fatal.append("dates_not_increasing")
    gaps = [(b - a).days for a, b in zip(d, d[1:])]
    if gaps and max(gaps) > max_gap_days:
        fatal.append(f"gap:{max(gaps)}d")
    r = close[1:] / close[:-1] - 1.0
    if np.abs(r).max() > max_abs_ret:
        fatal.append(f"extreme_return:{np.abs(r).max():.2f}")
    now = _dt.date.fromisoformat(today) if today else _dt.date.today()
    if (now - d[-1]).days > max_stale_days:
        fatal.append(f"stale:{(now - d[-1]).days}d")
    zero = int((vol <= 0).sum())
    if zero:
        warn.append(f"zero_volume_bars:{zero}")
    return fatal, warn


def _urllib_transport(url: str, headers: dict) -> dict:
    if not url.startswith(DATA_HOST + "/"):
        raise ValueError("refusing to fetch from an unexpected host")
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


class AlpacaDataFetcher:
    """Incremental daily-bar cache (CSV per symbol). Free IEX feed by default."""

    def __init__(self, cache_dir: str, key: str | None = None, secret: str | None = None,
                 transport=None, feed: str = "iex"):
        self.cache_dir = cache_dir
        self.key = key or os.environ.get("ALPACA_KEY")
        self.secret = secret or os.environ.get("ALPACA_SECRET")
        self.transport = transport or _urllib_transport
        self.feed = feed

    def path(self, symbol: str) -> str:
        return os.path.join(self.cache_dir, f"{symbol}.csv")

    def _cached(self, symbol: str) -> dict:
        p = self.path(symbol)
        out = {}
        if os.path.exists(p):
            with open(p) as f:
                for row in csv.DictReader(f):
                    out[row["date"]] = row
        return out

    def update(self, symbol: str, start: str = "2016-01-01", end: str | None = None) -> int:
        """Fetch bars newer than the cache. Returns the number of NEW bars."""
        if not (self.key and self.secret):
            raise RuntimeError("missing ALPACA_KEY / ALPACA_SECRET")
        have = self._cached(symbol)
        if have:
            last = max(have)
            start = (_dt.date.fromisoformat(last) + _dt.timedelta(days=1)).isoformat()
        headers = {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret}
        new, token = 0, None
        while True:
            url = (f"{DATA_HOST}/v2/stocks/{symbol}/bars?timeframe=1Day&start={start}&limit=10000"
                   f"&adjustment=all&feed={self.feed}&sort=asc")
            if end:
                url += f"&end={end}"
            if token:
                url += f"&page_token={token}"
            resp = self.transport(url, headers)
            for b in resp.get("bars") or []:
                day = b["t"][:10]
                if day not in have:
                    new += 1
                have[day] = {"date": day, "open": b["o"], "high": b["h"], "low": b["l"],
                             "close": b["c"], "volume": b["v"]}
            token = resp.get("next_page_token")
            if not token:
                break
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(self.path(symbol), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["date", "open", "high", "low", "close", "volume"])
            w.writeheader()
            for day in sorted(have):
                w.writerow({k: have[day][k] for k in w.fieldnames})
        return new

    def load(self, symbol: str) -> dict:
        return load_bars_csv(self.path(symbol))
