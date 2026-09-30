"""StrategyGenome: a structured, validated strategy representation (no free-form code)."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import hashlib
import json
from .primitives import PRIMITIVES

OPS = (">", "<")
SIDES = ("long", "short")


@dataclass
class Condition:
    primitive: str
    window: int
    op: str
    threshold: float

    def key(self) -> str:
        return f"{self.primitive}({self.window}){self.op}{self.threshold:.4f}"


@dataclass
class StrategyGenome:
    market: str = "SYNTH"
    timeframe: str = "1d"
    side: str = "long"
    entry: list = field(default_factory=list)
    exit: list = field(default_factory=list)
    filters: list = field(default_factory=list)
    position_size: float = 1.0     # fraction of the risk-permitted size
    stop_loss: float = 0.05        # fractional adverse move from entry
    max_hold: int = 20             # bars

    def complexity(self) -> int:
        return len(self.entry) + len(self.exit) + len(self.filters)

    def family(self) -> str:
        """Family = which primitives (plus side) are used, ignoring parameters."""
        prims = sorted({c.primitive for c in self.entry + self.filters})
        return self.side + ":" + "+".join(prims)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "StrategyGenome":
        g = StrategyGenome(**{k: v for k, v in d.items() if k not in ("entry", "exit", "filters")})
        g.entry = [Condition(**c) for c in d.get("entry", [])]
        g.exit = [Condition(**c) for c in d.get("exit", [])]
        g.filters = [Condition(**c) for c in d.get("filters", [])]
        return g

    def fingerprint(self) -> str:
        return hashlib.sha1(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:12]


def validate(g: StrategyGenome) -> list:
    """Return a list of problems; empty means valid."""
    errs = []
    if g.side not in SIDES:
        errs.append(f"bad side {g.side}")
    if not g.entry:
        errs.append("no entry conditions")
    if len(g.entry) > 5:
        errs.append("too many entry conditions")
    if not (0.0 < g.position_size <= 1.0):
        errs.append("position_size out of (0,1]")
    if not (0.002 <= g.stop_loss <= 0.5):
        errs.append("stop_loss out of range")
    if not (1 <= g.max_hold <= 250):
        errs.append("max_hold out of range")
    for c in g.entry + g.exit + g.filters:
        p = PRIMITIVES.get(c.primitive)
        if p is None:
            errs.append(f"unknown primitive {c.primitive}")
            continue
        if c.op not in OPS:
            errs.append(f"bad op {c.op}")
        if c.window not in p.windows:
            errs.append(f"{c.primitive} window {c.window} not allowed")
        lo, hi = p.bounds
        if not (lo <= c.threshold <= hi):
            errs.append(f"{c.primitive} threshold {c.threshold} outside [{lo},{hi}]")
    return errs
