import random
import unittest
import numpy as np

from rufflux.config import LabConfig, RiskLimits
from rufflux.data import synthetic_market, slice_bars
from rufflux.evaluation import (dominates, learning_efficiency, max_drawdown, objective_vector, pareto_rank,
                                total_return)
from rufflux.evolution import (add_condition, crossover, make_child, mutate_params, random_genome,
                               remove_condition, simplify)
from rufflux.execution import (ExecutionRefused, LiveExecution, LiveTradingDisabled, Order, PaperAlpaca)
from rufflux.genome import Condition, StrategyGenome, validate
from rufflux.governor import AccountState, RiskGovernor, RiskRequest
from rufflux.lab import Lab, evaluate_genome, next_stage
from rufflux.memory import Memory
from rufflux.novelty import novelty
from rufflux.sim import simulate
from rufflux.agents import genome_from_json, seed_hypotheses
import json


def acct(**k):
    d = dict(capital=10000.0, peak_capital=10000.0, day_start_capital=10000.0)
    d.update(k)
    return AccountState(**d)


class TestGenome(unittest.TestCase):
    def test_seed_hypotheses_valid(self):
        for h in seed_hypotheses():
            self.assertEqual(validate(h.genome), [], h.text)

    def test_random_genomes_valid_and_deterministic(self):
        a = [random_genome(random.Random(3)).fingerprint() for _ in range(3)]
        self.assertEqual(len(set(a)), 1)
        rng = random.Random(1)
        for _ in range(50):
            self.assertEqual(validate(random_genome(rng)), [])

    def test_validation_rejects_bad(self):
        g = StrategyGenome(entry=[Condition("nope", 20, ">", 0.0)])
        self.assertTrue(validate(g))
        g = StrategyGenome(entry=[Condition("momentum", 13, ">", 0.0)])
        self.assertTrue(validate(g))
        self.assertTrue(validate(StrategyGenome()))  # no entry

    def test_roundtrip_and_llm_gate(self):
        g = random_genome(random.Random(5))
        self.assertEqual(StrategyGenome.from_dict(g.to_dict()).fingerprint(), g.fingerprint())
        self.assertIsNotNone(genome_from_json(json.dumps(g.to_dict())))
        self.assertIsNone(genome_from_json("{not json"))
        bad = g.to_dict(); bad["stop_loss"] = 9.0
        self.assertIsNone(genome_from_json(json.dumps(bad)))


class TestMutation(unittest.TestCase):
    def test_all_operators_yield_valid_or_none(self):
        rng = random.Random(2)
        a, b = random_genome(rng), random_genome(rng)
        for op in ("params", "add", "remove", "simplify", "crossover", "radical"):
            for _ in range(30):
                c = make_child(op, a, b, rng)
                if c is not None:
                    self.assertEqual(validate(c), [], op)

    def test_mutation_reproducible_and_parent_untouched(self):
        g = random_genome(random.Random(9))
        before = g.fingerprint()
        c1 = mutate_params(g, random.Random(4))
        c2 = mutate_params(g, random.Random(4))
        self.assertEqual(c1.fingerprint(), c2.fingerprint())
        self.assertEqual(g.fingerprint(), before)

    def test_remove_keeps_one_entry_and_simplify_shrinks(self):
        g = StrategyGenome(entry=[Condition("momentum", 20, ">", 0.01)])
        self.assertIsNone(remove_condition(g, random.Random(1)))
        g.exit = [Condition("zscore", 20, ">", 0.0)]
        self.assertEqual(simplify(g, random.Random(1)).exit, [])


class TestSimAndFitness(unittest.TestCase):
    def setUp(self):
        self.bars = synthetic_market(800, seed=11)

    def test_sim_deterministic(self):
        g = seed_hypotheses()[0].genome
        r1 = simulate(g, self.bars).returns
        r2 = simulate(g, self.bars).returns
        np.testing.assert_array_equal(r1, r2)

    def test_no_lookahead_first_bar_flat(self):
        g = seed_hypotheses()[2].genome
        res = simulate(g, self.bars)
        self.assertEqual(res.positions[0], 0.0)

    def test_costs_reduce_returns(self):
        g = seed_hypotheses()[1].genome
        free = simulate(g, self.bars, 0, 0).returns.sum()
        paid = simulate(g, self.bars, 5, 5).returns.sum()
        self.assertLessEqual(paid, free)

    def test_drawdown_and_return_math(self):
        r = np.array([0.1, -0.1, 0.0])
        self.assertAlmostEqual(total_return(r), 1.1 * 0.9 - 1.0)
        self.assertAlmostEqual(max_drawdown(r), 0.1)
        self.assertEqual(max_drawdown(np.array([0.01, 0.01])), 0.0)

    def test_pareto(self):
        self.assertTrue(dominates([1, 1], [1, 0]))
        self.assertFalse(dominates([1, 0], [0, 1]))
        self.assertEqual(pareto_rank([[1, 1], [0, 0], [2, 0]]), [0, 1, 0])

    def test_learning_efficiency_prefers_cheap_informative(self):
        # surprising outcome at low risk beats an expected outcome at high risk
        cheap = learning_efficiency(1, 9, True, 0.005, 0.5)   # surprising win, tiny risk
        costly = learning_efficiency(9, 1, True, 0.20, 0.5)   # expected win, big risk
        self.assertGreater(cheap, costly)

    def test_objectives_not_profit_only(self):
        base = dict(m := {"return": 0.5, "drawdown": 0.0, "consistency": 1, "regime_stability": 1,
                          "simplicity": 0, "turnover": 0, "learning_efficiency": 1})
        risky = dict(base, drawdown=0.6, consistency=0.3)
        self.assertTrue(dominates(objective_vector(base), objective_vector(risky)))


class TestNovelty(unittest.TestCase):
    def test_small_archive_neutral(self):
        self.assertEqual(novelty(np.zeros(6), [])["score"], 0.5)

    def test_familiar_vs_unusual(self):
        rng = np.random.default_rng(0)
        archive = [list(rng.normal(0, 1, 6)) for _ in range(200)]
        fam = novelty(np.zeros(6), archive)["score"]
        odd = novelty(np.full(6, 6.0), archive)["score"]
        self.assertLess(fam, 0.5)
        self.assertGreater(odd, 0.9)
        self.assertGreater(odd, fam)

    def test_combination_novelty(self):
        # two features perfectly correlated in history; each value is common alone but the pair is not
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 300)
        archive = [[v, v, 0, 0, 0, 0] for v in x]
        together = novelty(np.array([1.0, 1.0, 0, 0, 0, 0]), archive)["score"]
        apart = novelty(np.array([1.5, -1.5, 0, 0, 0, 0]), archive)["score"]
        self.assertGreater(apart, together)


class TestMemoryAndLineage(unittest.TestCase):
    def test_lineage_and_records(self):
        mem = Memory()
        g = random_genome(random.Random(1))
        a = mem.add_strategy(g, [], 0, "seed")
        b = mem.add_strategy(g, [a], 1, "params")
        c = mem.add_strategy(g, [b, a], 2, "crossover")
        self.assertEqual(mem.lineage(c), [a, b, c])
        self.assertEqual(mem.lineage(a), [a])

    def test_experiment_tracking_and_similar_states(self):
        mem = Memory()
        g = random_genome(random.Random(1))
        sid = mem.add_strategy(g, [], 0, "seed")
        st = mem.add_state(0, [0.1] * 6, 0.5, "x")
        for win in (1, 1, 0):
            mem.add_experiment(sid, 0, st, g.family(), "h", {"m": 1}, win, 1.0, "lesson")
        self.assertEqual(mem.family_record(g.family()), (2, 1))
        rep = mem.similar_state_report([0.1] * 6, k=3, family=g.family())
        self.assertEqual((rep["wins"], rep["losses"]), (2, 1))
        self.assertEqual(mem.counts()["experiments"], 3)


class TestGovernorAndExecution(unittest.TestCase):
    def test_clipping_by_limits(self):
        gov = RiskGovernor(RiskLimits())
        d = gov.approve(RiskRequest(1, 5000.0), acct())
        self.assertEqual(d.permitted_notional, 1000.0)  # 10% of 10k
        d = gov.approve(RiskRequest(1, 500.0), acct())
        self.assertEqual(d.permitted_notional, 500.0)
        d = gov.approve(RiskRequest(1, 900.0), acct(experimental_notional=2000.0))
        self.assertEqual(d.permitted_notional, 500.0)   # 25% cap minus used

    def test_max_simultaneous(self):
        gov = RiskGovernor()
        self.assertEqual(gov.approve(RiskRequest(1, 100), acct(open_experiments=5)).permitted_notional, 0.0)

    def test_drawdown_halt_latches(self):
        gov = RiskGovernor()
        s = acct(capital=8900.0)
        d = gov.approve(RiskRequest(1, 100), s)
        self.assertTrue(d.halted)
        # recovering capital does not clear a halt
        self.assertTrue(gov.approve(RiskRequest(1, 100), acct()).halted)
        with self.assertRaises(PermissionError):
            gov.operator_reset("please")
        gov.operator_reset("I-AM-THE-HUMAN-OPERATOR")
        self.assertFalse(gov.approve(RiskRequest(1, 100), acct()).halted)

    def test_other_halts(self):
        for kw in ({"data_age_s": 10_000.0}, {"last_slippage_bps": 200.0}, {"capital": 9700.0}):
            gov = RiskGovernor()
            self.assertTrue(gov.approve(RiskRequest(1, 100), acct(**kw)).halted, kw)
        gov = RiskGovernor(); gov.emergency_shutdown()
        self.assertTrue(gov.halted)

    def test_limits_are_frozen(self):
        with self.assertRaises(Exception):
            RiskLimits().max_drawdown = 1.0
        gov = RiskGovernor()
        with self.assertRaises(AttributeError):
            gov.limits = RiskLimits(max_drawdown=1.0)

    def test_live_is_disabled(self):
        with self.assertRaises(LiveTradingDisabled):
            LiveExecution()
        self.assertFalse(LabConfig().live_trading_enabled)

    def test_paper_refuses_non_paper_and_no_governor(self):
        with self.assertRaises(ExecutionRefused):
            PaperAlpaca(RiskGovernor(), base_url="https://api.alpaca.markets")
        with self.assertRaises(ExecutionRefused):
            PaperAlpaca(None)

    def test_paper_routes_through_governor(self):
        sent = []
        ex = PaperAlpaca(RiskGovernor(), key="k", secret="s", transport=lambda p, b: sent.append(b) or {"ok": 1})
        ex.submit(Order(1, "SPY", "buy", 5000.0), acct())
        self.assertEqual(sent[0]["notional"], 1000.0)   # clipped by governor
        with self.assertRaises(ExecutionRefused):
            ex.submit(Order(1, "SPY", "buy", 100.0), acct(capital=8000.0))  # drawdown -> HALT
        self.assertEqual(len(sent), 1)

    def test_paper_needs_credentials(self):
        ex = PaperAlpaca(RiskGovernor(), key=None, secret=None, transport=lambda p, b: {})
        ex._key = ex._secret = None
        with self.assertRaises(ExecutionRefused):
            ex.submit(Order(1, "SPY", "buy", 100.0), acct())


class TestLab(unittest.TestCase):
    def test_lab_is_reproducible_and_never_touches_holdout(self):
        cfg = LabConfig(generations=4, population=12, survivors=4, db_path=":memory:")
        l1, l2 = Lab(cfg), Lab(cfg)
        s1, s2 = l1.run(), l2.run()
        strip = lambda s: [{k: v for k, v in x.items() if k != "top"} for x in s]
        self.assertEqual(strip(s1), strip(s2))
        self.assertEqual(l1.mem.counts()["holdout_peeks"], 0)
        # loop only ever sees dev bars
        self.assertLessEqual(3 * cfg.step + cfg.window, l1.n_dev)

    def test_descendants_link_to_parents(self):
        lab = Lab(LabConfig(generations=3, population=12, survivors=4, db_path=":memory:"))
        lab.run()
        rows = lab.mem.db.execute("SELECT id, parents FROM strategies WHERE generation>0").fetchall()
        self.assertTrue(rows)
        for r in rows:
            for p in json.loads(r["parents"]):
                self.assertLess(p, r["id"])

    def test_stage_machine(self):
        self.assertEqual(next_stage("UNKNOWN", True, False, 0, 0), "SIMULATED")
        self.assertEqual(next_stage("SIMULATED", True, True, 1, 1), "VALIDATION")
        self.assertEqual(next_stage("VALIDATION", True, True, 3, 3), "PAPER")
        self.assertEqual(next_stage("VALIDATION", True, True, 2, 2), "VALIDATION")

    def test_holdout_peek_is_logged(self):
        lab = Lab(LabConfig(generations=2, population=10, survivors=4, db_path=":memory:"))
        lab.run()
        lab.holdout_report(1)
        self.assertEqual(lab.mem.counts()["holdout_peeks"], 1)


if __name__ == "__main__":
    unittest.main()
