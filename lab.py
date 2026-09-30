"""The evolutionary loop: sense -> novelty -> hypothesize -> experiment -> evaluate -> learn ->
select -> mutate -> next generation. Simulation only in V0.1 (no orders are ever sent)."""
from __future__ import annotations
import argparse
import json
import logging
import random
import numpy as np

from .config import LabConfig, RiskLimits
from .data import synthetic_market, slice_bars
from .evaluation import (evaluate, learning_efficiency, max_drawdown, objective_vector, pareto_rank,
                         regime_labels, total_return, viable)
from .evolution import make_child, add_filter, random_genome
from .agents import seed_hypotheses, lesson_from, HIGH_VOL_FILTER, Hypothesis
from .genome import Condition, StrategyGenome
from .memory import Memory
from .novelty import state_vector, state_series, novelty
from .primitives import compute, rolling_std
from .sim import simulate

log = logging.getLogger("rufflux")
WARMUP = 100
STAGES = ("UNKNOWN", "SIMULATED", "VALIDATION", "PAPER", "CONTROLLED", "ESTABLISHED")


class SurvivalIntelligence:
    """Tracks the resources that keep the lab able to keep learning, and scales exploration."""

    def __init__(self, budget: float = 1.0):
        self.budget = budget

    def consume(self, risk: float) -> None:
        self.budget = max(0.0, self.budget - 0.1 * risk)

    def recover(self) -> None:
        self.budget = min(1.0, self.budget + 0.05)

    def exploration_scale(self) -> float:
        return 0.3 + 0.7 * self.budget


def _trades(pos: np.ndarray) -> int:
    return int(((pos[1:] != 0) & (pos[:-1] == 0)).sum()) if len(pos) > 1 else 0


def evaluate_genome(g: StrategyGenome, bars_w: dict, cfg: LabConfig, cache: dict) -> dict:
    """Deterministic. Train stats on first 70% of the eval region; out-of-sample on the last 30%."""
    sim = simulate(g, bars_w, cfg.fee_bps, cfg.slippage_bps, cache)
    vol20 = rolling_std(np.concatenate([[0.0], bars_w["close"][1:] / bars_w["close"][:-1] - 1.0]), 20)
    lab = regime_labels(vol20)[WARMUP:]
    r, pos = sim.returns[WARMUP:], sim.positions[WARMUP:]
    split = int(len(r) * 0.7)
    tr, va, pos_tr, pos_va = r[:split], r[split:], pos[:split], pos[split:]
    turn_tr = float(np.abs(np.diff(pos_tr, prepend=0.0)).sum())
    turn_va = float(np.abs(np.diff(pos_va, prepend=0.0)).sum())
    m = evaluate(tr, pos_tr, _trades(pos_tr), turn_tr, g.complexity(), lab[:split], cfg.n_folds, cfg.complexity_weight)
    m["train_return"] = m["return"]
    m["val_return"] = total_return(va)
    m["val_drawdown"] = max_drawdown(va)
    m["val_trades"] = _trades(pos_va)
    m["cost_val"] = turn_va * (cfg.fee_bps + cfg.slippage_bps) / 1e4
    m["return"] = m["val_return"]          # selection uses out-of-sample return
    m["drawdown"] = max(m["drawdown"], m["val_drawdown"])
    m["risk_consumed"] = m["val_drawdown"] + m["cost_val"]
    m["win"] = bool(m["val_trades"] >= 3 and m["val_return"] > 0)
    return m


def next_stage(stage: str, viable_: bool, win: bool, wins: int, windows_won: int) -> str:
    if stage == "UNKNOWN":
        return "SIMULATED" if viable_ else "UNKNOWN"
    if stage == "SIMULATED":
        return "VALIDATION" if win else "SIMULATED"
    if stage == "VALIDATION":
        return "PAPER" if (wins >= 3 and windows_won >= 3) else "VALIDATION"
    return stage  # CONTROLLED / ESTABLISHED are not reachable in V0.1


class Lab:
    def __init__(self, cfg: LabConfig | None = None, bars: dict | None = None, mem: Memory | None = None,
                 validate_span: bool = True):
        self.cfg = cfg or LabConfig()
        self.rng = random.Random(self.cfg.seed)
        self.bars = bars if bars is not None else synthetic_market(3000, seed=self.cfg.seed)
        self.mem = mem or Memory(self.cfg.db_path)
        self.survival = SurvivalIntelligence()
        self.n_dev = len(self.bars["close"]) - self.cfg.holdout_bars
        need = self.cfg.window + self.cfg.generations * self.cfg.step
        self.holdout_range = (self.n_dev, len(self.bars["close"]))
        self.manage_hist = True       # False when a subclass supplies the market-memory itself
        if validate_span and self.n_dev < need:
            raise ValueError(f"not enough development data ({self.n_dev} < {need})")
        self.pop: list = []           # dicts: sid, genome, hypothesis
        self.lessons: dict = {}       # sid -> tag
        self.wins: dict = {}          # sid -> [generation,...] of winning experiments
        self.seen: set = set()
        self.hist_states: list = []   # sampled past states (market memory for novelty)

    # ------------------------------------------------------------------ population
    def _register(self, g, parents, gen, origin, hypothesis) -> dict | None:
        fp = g.fingerprint()
        if fp in self.seen:
            return None
        self.seen.add(fp)
        sid = self.mem.add_strategy(g, parents, gen, origin)
        self.mem.add_hypothesis(hypothesis, sid, gen)
        return {"sid": sid, "genome": g, "hypothesis": hypothesis}

    def _initial_population(self) -> list:
        pop = []
        for h in seed_hypotheses():
            e = self._register(h.genome, [], 0, "seed", h.text)
            if e:
                pop.append(e)
        while len(pop) < self.cfg.population:
            e = self._register(random_genome(self.rng), [], 0, "random", "Exploratory random strategy.")
            if e:
                pop.append(e)
        return pop

    def _breed(self, survivors: list, gen: int, nov: float, counters: dict) -> list:
        pop = list(survivors)
        # 1) lesson-driven descendants (failure -> knowledge -> targeted children)
        for s in survivors:
            if len(pop) >= self.cfg.population:
                break
            if self.lessons.get(s["sid"]) == "fails_high_vol" and not s["genome"].filters:
                for tag, cond in (("exclude_high_vol", HIGH_VOL_FILTER),
                                  ("test_high_vol", Condition("vol_regime", 100, ">", 0.66))):
                    c = add_filter(s["genome"], cond)
                    if c and len(pop) < self.cfg.population:
                        e = self._register(c, [s["sid"]], gen, "lesson:" + tag,
                                           f"{s['hypothesis']} [lesson: {tag}]")
                        if e:
                            pop.append(e)
                            counters["mutations"] += 1
        # 2) operator-driven descendants
        explore = self.survival.exploration_scale() * (0.5 + nov)
        weights = {"params": 4.0, "add": 2.0, "remove": 1.5, "simplify": 1.5, "crossover": 2.0, "radical": 1.0 * explore}
        ops, w = list(weights), list(weights.values())
        tries = 0
        while len(pop) < self.cfg.population and tries < 500:
            tries += 1
            a = survivors[tries % len(survivors)]
            b = self.rng.choice(survivors)
            op = self.rng.choices(ops, w)[0]
            child = make_child(op, a["genome"], b["genome"], self.rng)
            if child is None:
                continue
            parents = [a["sid"], b["sid"]] if op == "crossover" else [a["sid"]]
            hyp = a["hypothesis"] if op != "radical" else "Radically different exploratory strategy."
            e = self._register(child, parents, gen, op, f"{hyp} [{op}]")
            if e:
                pop.append(e)
                counters["crossovers" if op == "crossover" else "mutations"] += 1
        return pop

    # ------------------------------------------------------------------ one generation
    def run_generation(self, gen: int, window: tuple | None = None) -> dict:
        cfg = self.cfg
        a, b = window if window else (gen * cfg.step, gen * cfg.step + cfg.window)
        bars_w = slice_bars(self.bars, a, b)
        counters = {"mutations": 0, "crossovers": 0}

        # sensors + novelty
        state = state_vector(bars_w)
        nv = novelty(state, self.mem.state_archive() + [list(v) for v in self.hist_states])
        pct = compute("vol_regime", 100, bars_w)[-1]
        regime = "high-vol" if pct > 0.66 else "low-vol" if pct < 0.33 else "mid-vol"
        state_id = self.mem.add_state(gen, state, nv["score"], nv["reason"], regime)
        # grow market memory with only the newly revealed bars (avoids re-adding old states)
        if self.manage_hist:
            fresh = WARMUP if gen == 0 else cfg.window - cfg.step
            self.hist_states.extend(state_series(bars_w, fresh, every=10))
        log.info("gen %d | novelty %.2f | regime %s | %s", gen, nv["score"], regime, nv["reason"])
        if nv["score"] > 0.7:
            self.mem.log(gen, "investigate", f"High novelty {nv['score']:.2f}: {nv['reason']}")

        # population
        if gen == 0 and not self.pop:
            self.pop = self._initial_population()
        # (later generations: self.pop was built at the end of the previous generation)

        # experiments
        cache: dict = {}
        results = []
        for e in self.pop:
            m = evaluate_genome(e["genome"], bars_w, cfg, cache)
            fam = e["genome"].family()
            w_, l_ = self.mem.family_record(fam)
            m["learning_efficiency"] = learning_efficiency(1 + w_, 1 + l_, m["win"], m["risk_consumed"], nv["score"])
            lesson_text, tag = lesson_from(m, m["win"])
            self.lessons[e["sid"]] = tag
            self.mem.add_experiment(e["sid"], gen, state_id, fam, e["hypothesis"], m, m["win"],
                                    m["learning_efficiency"], f"{tag}: {lesson_text}")
            self.survival.consume(m["risk_consumed"])
            if m["win"]:
                self.wins.setdefault(e["sid"], []).append(gen)
            results.append((e, m))

        # selection (multi-objective; profit alone never decides)
        cand = [(e, m) for e, m in results if viable(m)]
        vecs = [objective_vector(m) for _, m in cand]
        ranks = pareto_rank(vecs) if vecs else []
        order = sorted(range(len(cand)), key=lambda i: (ranks[i], -sum(vecs[i])))
        keep = {cand[i][0]["sid"] for i in order[: cfg.survivors]}
        survivors, discarded = [], 0
        for e, m in results:
            row = self.mem.get_strategy(e["sid"])
            wins = self.wins.get(e["sid"], [])
            stage = next_stage(row["stage"], viable(m), m["win"], len(wins), len(set(wins)))
            if e["sid"] in keep:
                survivors.append(e)
                self.mem.set_status(e["sid"], "alive", stage)
            else:
                discarded += 1
                why = "inactive (too few trades)" if not viable(m) else "dominated in multi-objective selection"
                self.mem.set_status(e["sid"], "discarded", stage, f"gen {gen}: {why}")
        if not survivors:  # never let the lab go extinct: reseed from scratch
            survivors = [self._register(random_genome(self.rng), [], gen, "reseed", "Extinction reseed.") or self.pop[0]]

        # evolve (next generation is built now so the summary can report it)
        top = sorted(results, key=lambda r: (r[0]["sid"] not in keep, -r[1]["learning_efficiency"]))[:3]
        top_ids = [e["sid"] for e, _ in top]
        if gen < cfg.generations - 1:
            self.pop = self._breed(survivors, gen + 1, nv["score"], counters)
        self.survival.recover()

        summary = {"generation": gen, "strategies": len(results), "survivors": len(survivors),
                   "experiments": len(results), "discarded": discarded, **counters,
                   "novelty": nv["score"], "novelty_reason": nv["reason"], "regime": regime,
                   "top": top_ids, "risk_budget": round(self.survival.budget, 3)}
        self.mem.log(gen, "generation_summary", json.dumps(summary))
        return summary

    def run(self) -> list:
        return [self.run_generation(g) for g in range(self.cfg.generations)]

    # ------------------------------------------------------------------ sealed holdout
    def holdout_report(self, sid: int) -> dict:
        """The ONLY code path that touches the sealed tail. Every peek is logged."""
        self.mem.log_holdout_peek(sid)
        g = StrategyGenome.from_dict(json.loads(self.mem.get_strategy(sid)["genome"]))
        h0, h1 = self.holdout_range
        a = h0 - WARMUP
        bars_h = slice_bars(self.bars, a, h1)
        sim = simulate(g, bars_h, self.cfg.fee_bps, self.cfg.slippage_bps)
        r, pos = sim.returns[WARMUP:], sim.positions[WARMUP:]
        return {"return": total_return(r), "drawdown": max_drawdown(r), "trades": _trades(pos)}


def main(argv=None):
    ap = argparse.ArgumentParser(description="RUFFLUX evolution lab (V0.1, simulation only)")
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--population", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--db", default="rufflux.db")
    ap.add_argument("--out", default="docs/index.html")
    ap.add_argument("--holdout", action="store_true", help="peek the sealed holdout for the top strategy (logged)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    cfg = LabConfig(seed=a.seed, population=a.population, generations=a.generations, db_path=a.db)
    lab = Lab(cfg)
    out = lab.run()
    from .dashboard import write_dashboard
    write_dashboard(lab.mem, cfg, RiskLimits(), a.out)
    last = out[-1]
    print(f"done: {last['strategies']} strategies in final generation, top ids {last['top']}, dashboard -> {a.out}")
    if a.holdout and last["top"]:
        print("sealed holdout (logged peek):", lab.holdout_report(last["top"][0]))


if __name__ == "__main__":
    main()
