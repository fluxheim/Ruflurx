"""Evolutionist: mutation, crossover, simplification. All randomness flows through a seeded
random.Random so evolution is reproducible. Every operator returns a VALID genome or None."""
from __future__ import annotations
import copy
import random
from .genome import Condition, StrategyGenome, validate
from .primitives import PRIMITIVES

# Sensible sampling ranges per primitive (extend when registering new primitives).
TYPICAL = {
    "returns": (-0.02, 0.02), "momentum": (-0.08, 0.08), "trend": (-0.05, 0.05),
    "volatility": (0.004, 0.025), "volume_z": (-1.5, 1.5), "breakout_high": (-0.06, 0.0),
    "breakdown_low": (0.0, 0.06), "zscore": (-2.0, 2.0), "vol_regime": (0.1, 0.9), "dow": (0.5, 3.5),
}
NON_ENTRY = {"dow"}  # primitives kept out of random entry rules (still usable as filters)


def _range(name: str):
    return TYPICAL.get(name, PRIMITIVES[name].bounds)


def random_condition(rng: random.Random, allow_regime: bool = False) -> Condition:
    names = [n for n in PRIMITIVES if n not in NON_ENTRY and (allow_regime or n != "vol_regime")]
    name = rng.choice(sorted(names))
    p = PRIMITIVES[name]
    lo, hi = _range(name)
    return Condition(name, rng.choice(p.windows), rng.choice([">", "<"]), round(rng.uniform(lo, hi), 4))


def random_genome(rng: random.Random) -> StrategyGenome:
    g = StrategyGenome(side=rng.choice(["long", "short"]),
                       stop_loss=round(rng.choice([0.02, 0.03, 0.05, 0.08]), 3),
                       max_hold=rng.choice([5, 10, 20, 40]),
                       position_size=rng.choice([0.5, 1.0]))
    g.entry = [random_condition(rng) for _ in range(rng.choice([1, 2, 2, 3]))]
    if rng.random() < 0.3:
        g.exit = [random_condition(rng)]
    return g if not validate(g) else random_genome(rng)


def _clip(c: Condition) -> Condition:
    lo, hi = PRIMITIVES[c.primitive].bounds
    c.threshold = round(min(max(c.threshold, lo), hi), 4)
    return c


def mutate_params(g: StrategyGenome, rng: random.Random, scale: float = 0.15) -> StrategyGenome | None:
    n = copy.deepcopy(g)
    for c in n.entry + n.exit + n.filters:
        lo, hi = _range(c.primitive)
        c.threshold += rng.gauss(0, scale * (hi - lo))
        _clip(c)
        if rng.random() < 0.15:
            c.window = rng.choice(PRIMITIVES[c.primitive].windows)
    n.stop_loss = round(min(max(n.stop_loss * rng.choice([0.7, 1.0, 1.4]), 0.005), 0.2), 3)
    n.max_hold = int(min(max(n.max_hold + rng.choice([-5, 0, 5]), 2), 100))
    return n if not validate(n) else None


def add_condition(g: StrategyGenome, rng: random.Random) -> StrategyGenome | None:
    n = copy.deepcopy(g)
    if len(n.entry) >= 4:
        return None
    n.entry.append(random_condition(rng))
    return n if not validate(n) else None


def remove_condition(g: StrategyGenome, rng: random.Random) -> StrategyGenome | None:
    n = copy.deepcopy(g)
    pool = [("entry", i) for i in range(len(n.entry))] if len(n.entry) > 1 else []
    pool += [("exit", i) for i in range(len(n.exit))] + [("filters", i) for i in range(len(n.filters))]
    if not pool:
        return None
    part, i = rng.choice(pool)
    getattr(n, part).pop(i)
    return n if not validate(n) else None


def simplify(g: StrategyGenome, rng: random.Random) -> StrategyGenome | None:
    """Drop exit rules and filters first, then one entry condition. Selection judges the result."""
    n = copy.deepcopy(g)
    if n.exit:
        n.exit = []
    elif n.filters:
        n.filters.pop(rng.randrange(len(n.filters)))
    elif len(n.entry) > 1:
        n.entry.pop(rng.randrange(len(n.entry)))
    else:
        return None
    return n if not validate(n) else None


def add_filter(g: StrategyGenome, cond: Condition) -> StrategyGenome | None:
    n = copy.deepcopy(g)
    if any(f.key() == cond.key() for f in n.filters):
        return None
    n.filters.append(cond)
    return n if not validate(n) else None


def crossover(a: StrategyGenome, b: StrategyGenome, rng: random.Random) -> StrategyGenome | None:
    n = copy.deepcopy(a)
    pick = rng.choice(["entry", "filters", "exit"])
    n.entry = copy.deepcopy(a.entry)
    if pick == "entry" and b.entry:
        cut = rng.randint(1, len(b.entry))
        n.entry = copy.deepcopy(a.entry[: max(1, len(a.entry) // 2)] + b.entry[:cut])[:4]
    elif pick == "filters":
        n.filters = copy.deepcopy(b.filters)
    else:
        n.exit = copy.deepcopy(b.exit)
    return n if not validate(n) else None


OPERATORS = ("params", "add", "remove", "simplify", "crossover", "radical")


def make_child(op: str, a: StrategyGenome, b: StrategyGenome, rng: random.Random):
    if op == "params":
        return mutate_params(a, rng)
    if op == "add":
        return add_condition(a, rng)
    if op == "remove":
        return remove_condition(a, rng)
    if op == "simplify":
        return simplify(a, rng)
    if op == "crossover":
        return crossover(a, b, rng)
    if op == "radical":
        return random_genome(rng)
    raise ValueError(op)
