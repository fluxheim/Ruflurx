"""Primitive library. Add a primitive by registering it; genomes may then use it."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np


def rolling_mean(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        c = np.cumsum(np.insert(x, 0, 0.0))
        out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    m = rolling_mean(x, n)
    m2 = rolling_mean(x * x, n)
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def rolling_max_prior(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(n, len(x)):
        out[i] = x[i - n:i].max()
    return out


def rolling_min_prior(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(n, len(x)):
        out[i] = x[i - n:i].min()
    return out


def _returns(bars: dict) -> np.ndarray:
    c = bars["close"]
    r = np.zeros(len(c))
    r[1:] = c[1:] / c[:-1] - 1.0
    return r


def _lagret(bars: dict, n: int) -> np.ndarray:
    c = bars["close"]
    return np.concatenate([np.full(n, np.nan), c[n:] / c[:-n] - 1.0])


def _safe(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, x, np.nan)


def _percentile_rank(x: np.ndarray, n: int) -> np.ndarray:
    """Rank of x[t] within the trailing n observations (0..1). Causal."""
    out = np.full(len(x), np.nan)
    for i in range(n, len(x)):
        w = x[i - n + 1:i + 1]
        if np.isnan(w).any():
            continue
        out[i] = (w < x[i]).mean()
    return out


@dataclass(frozen=True)
class Primitive:
    name: str
    fn: Callable[[dict, int], np.ndarray]
    windows: tuple
    bounds: tuple  # allowed threshold range
    kind: str


PRIMITIVES: dict[str, Primitive] = {}


def register(p: Primitive) -> None:
    PRIMITIVES[p.name] = p


register(Primitive("returns", _lagret, (1, 3, 5, 10), (-0.1, 0.1), "returns"))
register(Primitive("momentum", _lagret, (10, 20, 40), (-0.3, 0.3), "momentum"))
register(Primitive("trend", lambda b, n: b["close"] / rolling_mean(b["close"], n) - 1.0,
                   (10, 20, 50), (-0.2, 0.2), "trend"))
register(Primitive("volatility", lambda b, n: rolling_std(_returns(b), n),
                   (10, 20, 40), (0.001, 0.08), "volatility"))
register(Primitive("volume_z", lambda b, n: (b["volume"] - rolling_mean(b["volume"], n)) /
                   _safe(rolling_std(b["volume"], n)), (10, 20, 40), (-3.0, 3.0), "volume"))
register(Primitive("breakout_high", lambda b, n: b["close"] / rolling_max_prior(b["close"], n) - 1.0,
                   (10, 20, 40), (-0.2, 0.1), "breakout"))
register(Primitive("breakdown_low", lambda b, n: b["close"] / rolling_min_prior(b["close"], n) - 1.0,
                   (10, 20, 40), (-0.1, 0.2), "breakout"))
register(Primitive("zscore", lambda b, n: (b["close"] - rolling_mean(b["close"], n)) /
                   _safe(rolling_std(b["close"], n)), (10, 20, 40), (-3.0, 3.0), "mean_reversion"))
register(Primitive("vol_regime", lambda b, n: _percentile_rank(rolling_std(_returns(b), 20), n),
                   (100,), (0.0, 1.0), "regime"))
register(Primitive("dow", lambda b, n: (np.arange(len(b["close"])) % 5).astype(float),
                   (1,), (0.0, 4.0), "time"))


def compute(name: str, window: int, bars: dict) -> np.ndarray:
    return PRIMITIVES[name].fn(bars, window)
