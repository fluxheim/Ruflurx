import csv
import json
import os
import tempfile
import unittest
import numpy as np

from rufflux.config import LabConfig
from rufflux.data import (AlpacaDataFetcher, _urllib_transport, check_quality, load_bars_csv, synthetic_market)
from rufflux.governor import RiskGovernor
from rufflux.memory import Memory
from rufflux.runner import HaltStore, ResumableLab, main


def dated_bars(n=1500, seed=3):
    b = synthetic_market(n, seed=seed)
    days = np.arange("2014-01-01", "2035-01-01", dtype="datetime64[D]")
    days = days[np.is_busday(days)][:n]
    b["dates"] = days.astype(str)
    return b


def write_csv(path, b):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "open", "high", "low", "close", "volume"])
        for d, c, v in zip(b["dates"], b["close"], b["volume"]):
            w.writerow([d, c, c, c, c, v])


def bar(day, c=100.0):
    return {"t": day + "T05:00:00Z", "o": c, "h": c, "l": c, "c": c, "v": 1000}


class TestFetcher(unittest.TestCase):
    def test_pagination_incremental_and_no_dupes(self):
        calls = []

        def transport(url, headers):
            calls.append(url)
            if "page_token=P2" in url:
                return {"bars": [bar("2024-01-03")], "next_page_token": None}
            if "start=2024-01-04" in url:
                return {"bars": [bar("2024-01-04"), bar("2024-01-05")], "next_page_token": None}
            return {"bars": [bar("2024-01-02")], "next_page_token": "P2"}

        with tempfile.TemporaryDirectory() as d:
            f = AlpacaDataFetcher(d, key="k", secret="s", transport=transport)
            self.assertEqual(f.update("SPY", start="2024-01-01"), 2)
            self.assertEqual(f.update("SPY"), 2)     # incremental: starts after the last cached day
            self.assertIn("start=2024-01-04", calls[-1])
            b = f.load("SPY")
            self.assertEqual(list(b["dates"]), ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])

    def test_needs_keys_and_host_locked(self):
        with tempfile.TemporaryDirectory() as d:
            f = AlpacaDataFetcher(d, key=None, secret=None, transport=lambda u, h: {})
            f.key = f.secret = None
            with self.assertRaises(RuntimeError):
                f.update("SPY")
        with self.assertRaises(ValueError):
            _urllib_transport("https://evil.example.com/x", {})


class TestQuality(unittest.TestCase):
    def setUp(self):
        self.b = dated_bars(600)
        self.today = str(self.b["dates"][-1])

    def test_clean_data_passes(self):
        fatal, _ = check_quality(self.b, today=self.today)
        self.assertEqual(fatal, [])

    def test_stale(self):
        fatal, _ = check_quality(self.b, today="2099-01-01")
        self.assertTrue(any(x.startswith("stale") for x in fatal))

    def test_gap_and_extreme_and_bad_prices(self):
        b = dated_bars(600)
        b["dates"] = b["dates"].copy(); b["dates"][300:] = np.array(
            [str(np.datetime64(x) + 30) for x in b["dates"][300:]])
        self.assertTrue(any(x.startswith("gap") for x in check_quality(b, today="2099-01-01")[0]))
        b = dated_bars(600); b["close"] = b["close"].copy(); b["close"][400] *= 3
        self.assertTrue(any(x.startswith("extreme") for x in check_quality(b, today=self.today)[0]))
        b = dated_bars(600); b["close"] = b["close"].copy(); b["close"][10] = -1
        self.assertIn("bad_prices", check_quality(b, today=self.today)[0])

    def test_too_few_bars(self):
        self.assertTrue(check_quality(dated_bars(100), today="2099-01-01")[0][0].startswith("too_few"))


class TestHaltStore(unittest.TestCase):
    def test_halt_persists_across_runs(self):
        with tempfile.TemporaryDirectory() as d:
            store = HaltStore(os.path.join(d, "halt.json"))
            g1 = RiskGovernor(); g1.data_quality_halt(["SPY:stale:9d"]); store.save(g1)
            g2 = RiskGovernor(); store.load_into(g2)
            self.assertTrue(g2.halted)
            self.assertIn("data_quality:SPY:stale:9d", g2.halt_reasons)


class TestResumableLab(unittest.TestCase):
    def cfg(self, path):
        return LabConfig(generations=10 ** 9, db_path=path)

    def test_resume_continues_generations_population_and_lineage(self):
        b = dated_bars(1500)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.db")
            lab = ResumableLab(self.cfg(path), b, Memory(path))
            s0 = lab.run_one(); s1 = lab.run_one()
            self.assertEqual((s0["generation"], s1["generation"]), (0, 1))
            n_strats = lab.mem.counts()["strategies"]
            lab.mem.db.close()
            lab2 = ResumableLab(self.cfg(path), b, Memory(path))   # a brand-new process
            self.assertEqual(lab2.next_gen, 2)
            self.assertGreaterEqual(len(lab2.pop), lab2.cfg.survivors)
            s2 = lab2.run_one()
            self.assertEqual(s2["generation"], 2)
            self.assertGreater(lab2.mem.counts()["strategies"], n_strats - 1)
            # lineage reaches back across runs
            newest = lab2.mem.db.execute("SELECT MAX(id) m FROM strategies WHERE generation>=2").fetchone()["m"]
            self.assertGreaterEqual(len(lab2.mem.lineage(newest)), 2)

    def test_windows_never_touch_sealed_holdout(self):
        b = dated_bars(1500)
        lab = ResumableLab(LabConfig(generations=10 ** 9, db_path=":memory:"), b, Memory())
        h0, h1 = lab.holdout_range
        for gen in range(40):
            a, e = lab.choose_window(gen)
            self.assertTrue(e <= h0 or a >= h1, (gen, a, e, h0, h1))
        # once enough new bars exist after the holdout, the latest window is used
        big = dated_bars(2300)
        lab2 = ResumableLab(LabConfig(generations=10 ** 9, db_path=":memory:"), big, Memory())
        h0, h1 = lab2.holdout_range
        lab2.holdout_range = (h0 - 800, h1 - 800)   # pretend holdout was sealed earlier
        a, e = lab2.choose_window(0)
        self.assertGreaterEqual(a, lab2.holdout_range[1])
        self.assertEqual(e, 2300)

    def test_holdout_dates_are_sealed_once(self):
        b = dated_bars(1500)
        m = Memory()
        ResumableLab(LabConfig(generations=10 ** 9, db_path=":memory:"), b, m)
        first = (m.get_meta("holdout_start"), m.get_meta("holdout_end"))
        longer = dated_bars(1600)
        lab = ResumableLab(LabConfig(generations=10 ** 9, db_path=":memory:"), longer, m)
        self.assertEqual((m.get_meta("holdout_start"), m.get_meta("holdout_end")), first)
        self.assertEqual(lab.mem.counts()["holdout_peeks"], 0)

    def test_skips_without_new_bars_in_live_window_mode(self):
        big = dated_bars(2300)
        m = Memory()
        m.set_meta("holdout_start", str(big["dates"][700])); m.set_meta("holdout_end", str(big["dates"][1199]))
        lab = ResumableLab(LabConfig(generations=10 ** 9, db_path=":memory:"), big, m)
        self.assertIsNotNone(lab.run_one())
        self.assertIsNone(lab.run_one())                 # same last bar -> skip
        self.assertIsNotNone(lab.run_one(force=True))    # manual override


class TestRunnerEndToEnd(unittest.TestCase):
    def run_main(self, d, *extra):
        return main(["--symbols", "SPY", "--offline", "--data-dir", os.path.join(d, "data"),
                     "--state-dir", os.path.join(d, "state"), "--out", os.path.join(d, "docs"), *extra])

    def test_offline_run_then_stale_halt_then_reset(self):
        b = dated_bars(1500)
        with tempfile.TemporaryDirectory() as d:
            write_csv(os.path.join(d, "data", "SPY.csv"), b)
            self.assertEqual(self.run_main(d, "--today", str(b["dates"][-1])), 0)
            self.assertTrue(os.path.exists(os.path.join(d, "docs", "SPY.html")))
            self.assertTrue(os.path.exists(os.path.join(d, "docs", "index.html")))
            self.assertFalse(json.load(open(os.path.join(d, "state", "halt.json")))["halted"])
            # data goes stale -> data-quality HALT, latched
            self.run_main(d, "--today", "2099-01-01")
            halt = json.load(open(os.path.join(d, "state", "halt.json")))
            self.assertTrue(halt["halted"])
            db = Memory(os.path.join(d, "state", "SPY.db"))
            gens = db.db.execute("SELECT MAX(generation) g FROM experiments").fetchone()["g"]
            # a later run with good data still refuses to run (HALT persists)
            self.run_main(d, "--today", str(b["dates"][-1]), "--force")
            db2 = Memory(os.path.join(d, "state", "SPY.db"))
            self.assertEqual(db2.db.execute("SELECT MAX(generation) g FROM experiments").fetchone()["g"], gens)
            self.assertIn("HALTED", open(os.path.join(d, "docs", "index.html")).read())
            # human operator reset
            self.run_main(d, "--today", str(b["dates"][-1]), "--reset-halt", "--force")
            self.assertFalse(json.load(open(os.path.join(d, "state", "halt.json")))["halted"])


if __name__ == "__main__":
    unittest.main()
