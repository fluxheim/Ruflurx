"""Scientist + Strategy Architect. V0.1 ships a deterministic, rule-based Scientist so the loop
runs offline. An LLM can be plugged in later via `genome_from_json`: the LLM only ever emits
JSON that must pass `validate`; it never writes executable code."""
from __future__ import annotations
import json
from dataclasses import dataclass
from .genome import Condition, StrategyGenome, validate


@dataclass(frozen=True)
class Hypothesis:
    text: str
    genome: StrategyGenome


def _c(p, w, op, t):
    return Condition(p, w, op, t)


def seed_hypotheses() -> list:
    """Starting hypotheses (not strategies to trust: things to test)."""
    return [
        Hypothesis("Volatility compression followed by abnormal volume predicts continuation.",
                   StrategyGenome(side="long", entry=[_c("volatility", 20, "<", 0.008), _c("volume_z", 20, ">", 1.0),
                                                      _c("momentum", 10, ">", 0.0)], max_hold=10, stop_loss=0.03)),
        Hypothesis("Sharp short-term overextension below the mean reverts.",
                   StrategyGenome(side="long", entry=[_c("zscore", 20, "<", -1.8)],
                                  exit=[_c("zscore", 20, ">", 0.0)], max_hold=10, stop_loss=0.04)),
        Hypothesis("Breakouts above the recent high persist when trend is already up.",
                   StrategyGenome(side="long", entry=[_c("breakout_high", 20, ">", -0.005), _c("trend", 50, ">", 0.0)],
                                  max_hold=20, stop_loss=0.05)),
        Hypothesis("Breakdowns below the recent low persist when trend is already down.",
                   StrategyGenome(side="short", entry=[_c("breakdown_low", 20, "<", 0.005), _c("trend", 50, "<", 0.0)],
                                  max_hold=20, stop_loss=0.05)),
        Hypothesis("Strong 20-bar momentum continues in low-volatility regimes.",
                   StrategyGenome(side="long", entry=[_c("momentum", 20, ">", 0.04)],
                                  filters=[_c("vol_regime", 100, "<", 0.6)], max_hold=15, stop_loss=0.04)),
        Hypothesis("Spikes in volume with negative returns mark exhaustion; fade them.",
                   StrategyGenome(side="long", entry=[_c("volume_z", 20, ">", 1.5), _c("returns", 3, "<", -0.01)],
                                  max_hold=5, stop_loss=0.03)),
    ]


def lesson_from(metrics: dict, win: bool) -> tuple:
    """Turn an outcome into (lesson_text, tag). Tags are machine-readable for descendants."""
    rb = metrics.get("regime_pnl", {})
    if not win and rb.get("high_vol", 0.0) < 0 <= min(rb.get("low_vol", 0.0), rb.get("mid_vol", 0.0)):
        return ("Relationship appeared unreliable during high-volatility regimes.", "fails_high_vol")
    if not win and metrics.get("turnover", 0) > 0.2:
        return ("Edge (if any) was eaten by trading costs.", "cost_drag")
    if not win:
        return ("No reliable edge out-of-sample.", "no_edge")
    if metrics.get("regime_stability", 0) < 0.67:
        return ("Worked out-of-sample but only in some regimes.", "regime_dependent")
    return ("Held up out-of-sample across regimes.", "robust")


HIGH_VOL_FILTER = Condition("vol_regime", 100, "<", 0.66)


def genome_from_json(text: str) -> StrategyGenome | None:
    """Parse an LLM-produced genome. Anything that fails validation is rejected, not repaired."""
    try:
        g = StrategyGenome.from_dict(json.loads(text))
    except Exception:
        return None
    return g if not validate(g) else None
