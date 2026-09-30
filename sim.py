"""Deterministic backtester. Signals at bar t act on the return of bar t+1 (no lookahead).
Costs (fee + slippage) are charged per unit of turnover."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .genome import StrategyGenome, Condition
from .primitives import compute


def _cond_mask(c: Condition, bars: dict, cache: dict) -> np.ndarray:
    k = (c.primitive, c.window)
    if k not in cache:
        cache[k] = compute(c.primitive, c.window, bars)
    x = cache[k]
    with np.errstate(invalid="ignore"):
        m = x > c.threshold if c.op == ">" else x < c.threshold
    return m & ~np.isnan(x)


def _all(conds, bars, cache, n):
    m = np.ones(n, dtype=bool)
    for c in conds:
        m &= _cond_mask(c, bars, cache)
    return m


@dataclass
class SimResult:
    returns: np.ndarray      # per-bar net strategy returns
    positions: np.ndarray    # per-bar position held over that bar
    n_trades: int
    turnover: float          # total units traded
    cost_paid: float         # total cost drag (fraction)


def simulate(g: StrategyGenome, bars: dict, fee_bps: float = 1.0, slippage_bps: float = 2.0,
             cache: dict | None = None) -> SimResult:
    close = bars["close"]
    n = len(close)
    cache = {} if cache is None else cache
    rets = np.zeros(n)
    rets[1:] = close[1:] / close[:-1] - 1.0
    entry = _all(g.entry, bars, cache, n) & _all(g.filters, bars, cache, n)
    exit_sig = _all(g.exit, bars, cache, n) if g.exit else np.zeros(n, dtype=bool)
    sign = 1.0 if g.side == "long" else -1.0
    unit = g.position_size
    pos = np.zeros(n)          # position held during bar t (decided at t-1)
    held, entry_px, bars_held = 0.0, 0.0, 0
    n_trades = 0
    for t in range(n - 1):
        cur = held
        if cur != 0.0:
            bars_held += 1
            adverse = sign * (close[t] / entry_px - 1.0)
            if exit_sig[t] or bars_held >= g.max_hold or adverse <= -g.stop_loss:
                cur = 0.0
        elif entry[t]:
            cur = sign * unit
            entry_px, bars_held = close[t], 0
            n_trades += 1
        if cur == 0.0:
            bars_held = 0
        held = cur
        pos[t + 1] = held
    trade_units = np.abs(np.diff(pos, prepend=0.0))
    cost = trade_units * (fee_bps + slippage_bps) / 1e4
    net = pos * rets - cost
    return SimResult(net, pos, n_trades, float(trade_units.sum()), float(cost.sum()))
