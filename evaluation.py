"""Evaluation lives OUTSIDE the generators. Agents cannot redefine success.
Multi-objective: no single 'profit' fitness. Ranking is Pareto-based."""
from __future__ import annotations
import math
import numpy as np

OBJECTIVES = ("return", "drawdown", "consistency", "regime_stability", "simplicity", "turnover", "learning_efficiency")


def max_drawdown(returns: np.ndarray) -> float:
    eq = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(np.concatenate([[1.0], eq]))[1:]
    return float(np.max(1.0 - eq / peak)) if len(eq) else 0.0


def sharpe(returns: np.ndarray) -> float:
    sd = returns.std()
    return float(returns.mean() / sd * math.sqrt(252)) if sd > 1e-12 else 0.0


def total_return(returns: np.ndarray) -> float:
    return float(np.prod(1.0 + returns) - 1.0)


def fold_bounds(n: int, k: int) -> list:
    edges = np.linspace(0, n, k + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(k)]


def regime_labels(vol20: np.ndarray) -> np.ndarray:
    """0 low / 1 mid / 2 high vol via terciles (of the given window)."""
    v = vol20[~np.isnan(vol20)]
    if len(v) < 10:
        return np.zeros(len(vol20), dtype=int)
    lo, hi = np.quantile(v, [1 / 3, 2 / 3])
    lab = np.where(vol20 < lo, 0, np.where(vol20 < hi, 1, 2))
    return np.where(np.isnan(vol20), 1, lab)


def regime_breakdown(returns: np.ndarray, labels: np.ndarray) -> dict:
    out = {}
    for r, name in ((0, "low_vol"), (1, "mid_vol"), (2, "high_vol")):
        m = labels == r
        out[name] = float(returns[m].sum()) if m.sum() > 5 else 0.0
    return out


def learning_efficiency(prior_a: float, prior_b: float, win: bool, risk_consumed: float,
                        novelty: float, eps: float = 0.01) -> float:
    """Conceptual objective: useful_information / resources_consumed.
    useful_information = surprise (bits) of the outcome under the family's Beta prior,
                         weighted up when the market state was novel.
    resources_consumed = risk actually consumed (drawdown + cost drag) plus a floor.
    Kept as a plain function so the formula can be swapped without touching agents."""
    p_win = prior_a / (prior_a + prior_b)
    p = p_win if win else 1.0 - p_win
    surprise = -math.log2(max(p, 1e-6))
    info = surprise * (1.0 + novelty)
    return info / (risk_consumed + eps)


def evaluate(sim_returns: np.ndarray, positions: np.ndarray, n_trades: int, turnover: float,
             complexity: int, regime_lab: np.ndarray, n_folds: int, complexity_weight: float) -> dict:
    n = len(sim_returns)
    folds = [sim_returns[a:b] for a, b in fold_bounds(n, n_folds)]
    fold_ret = [total_return(f) for f in folds]
    consistency = float(np.mean([r > 0 for r in fold_ret]))
    rb = regime_breakdown(sim_returns, regime_lab)
    regime_stability = float(np.mean([v > 0 for v in rb.values()]))
    dd = max_drawdown(sim_returns)
    tr = total_return(sim_returns)
    active = int((positions != 0).sum())
    return {
        "return": tr, "sharpe": sharpe(sim_returns), "drawdown": dd,
        "consistency": consistency, "worst_fold": float(min(fold_ret)),
        "regime_stability": regime_stability, "regime_pnl": rb,
        "turnover": turnover / max(n, 1), "n_trades": n_trades, "active_frac": active / max(n, 1),
        "complexity": complexity,
        "simplicity": -complexity_weight * complexity,
    }


def objective_vector(m: dict) -> list:
    """Higher is better for every element."""
    return [m["return"], -m["drawdown"], m["consistency"], m["regime_stability"],
            m["simplicity"], -m["turnover"], math.log1p(max(m.get("learning_efficiency", 0.0), 0.0))]


def dominates(a: list, b: list) -> bool:
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def pareto_rank(vectors: list) -> list:
    """Rank 0 = non-dominated. Tie-break elsewhere by crowding/simple sum."""
    ranks = [0] * len(vectors)
    remaining = set(range(len(vectors)))
    r = 0
    while remaining:
        front = [i for i in remaining if not any(dominates(vectors[j], vectors[i]) for j in remaining if j != i)]
        for i in front:
            ranks[i] = r
        remaining -= set(front)
        r += 1
    return ranks


def viable(m: dict) -> bool:
    """Minimum bar: enough activity to say anything at all."""
    return m["n_trades"] >= 5
