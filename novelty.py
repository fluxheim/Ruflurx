"""Novelty = unusual COMBINATION of familiar variables, measured by kNN distance in a
standardized state space against an archive of previously seen states."""
from __future__ import annotations
import numpy as np
from .primitives import compute

STATE_FEATURES = (
    ("returns", 5, "5-bar return"),
    ("volatility", 20, "volatility"),
    ("volume_z", 20, "volume"),
    ("momentum", 20, "momentum"),
    ("trend", 50, "trend"),
    ("breakout_high", 20, "breakout distance"),
)


def state_vector(bars: dict) -> np.ndarray:
    """Latest market state as a feature vector (causal, last bar)."""
    v = []
    for name, w, _ in STATE_FEATURES:
        x = compute(name, w, bars)
        v.append(float(x[-1]) if not np.isnan(x[-1]) else 0.0)
    return np.array(v)


def novelty(state: np.ndarray, archive: list, k: int = 5) -> dict:
    """Return {'score': 0..1, 'reason': str}. Score is 0.5 by convention for an empty archive."""
    labels = [f[2] for f in STATE_FEATURES]
    if len(archive) < max(k + 1, 8):
        return {"score": 0.5, "reason": "Archive too small to judge; treating state as moderately novel."}
    A = np.array(archive)
    mu, sd = A.mean(axis=0), A.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Z = (A - mu) / sd
    z = (state - mu) / sd
    d = np.linalg.norm(Z - z, axis=1)
    kk = min(k, len(d))
    d_k = float(np.sort(d)[:kk].mean())
    # reference: typical kNN distance within the archive itself (leave-one-out)
    ref = []
    for i in range(0, len(Z), max(1, len(Z) // 40)):
        di = np.linalg.norm(Z - Z[i], axis=1)
        di = np.sort(di)[1:kk + 1]
        ref.append(di.mean())
    d_ref = float(np.median(ref)) or 1.0
    score = float(1.0 - np.exp(-(d_k / d_ref) ** 2 * 0.5 * np.log(2) * 2))
    score = min(max(score, 0.0), 1.0)
    # reason: which features are individually familiar vs jointly unusual
    per = np.abs(z)
    order = np.argsort(-per)
    familiar = [labels[i] for i in range(len(labels)) if per[i] < 1.0]
    odd = [f"{'high' if z[i] > 0 else 'low'} {labels[i]}" for i in order[:2] if per[i] >= 1.0]
    if score > 0.6 and not odd:
        reason = ("Each variable looks familiar on its own (" + ", ".join(familiar[:3]) +
                  "), but the combination is unusual.")
    elif odd:
        reason = "Unusual combination driven by " + " and ".join(odd) + "."
    else:
        reason = "State resembles previously seen states."
    return {"score": score, "reason": reason}


def state_series(bars: dict, start: int, every: int = 10) -> list:
    """Sampled historical states (causal per bar) from `start` onward, used to grow the archive."""
    cols = [compute(name, w, bars) for name, w, _ in STATE_FEATURES]
    out = []
    for i in range(start, len(bars["close"]), every):
        row = [c[i] for c in cols]
        if not any(np.isnan(row)):
            out.append(np.array(row))
    return out
