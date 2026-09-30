"""Unattended daily runner (GitHub Actions). Pull real bars -> quality gate -> one generation per
symbol -> dashboards. Simulation only: no orders are placed anywhere in this module."""
from __future__ import annotations
import argparse
import json
import logging
import os
import random
import numpy as np

from .config import LabConfig, RiskLimits
from .data import AlpacaDataFetcher, check_quality, slice_bars
from .evolution import random_genome  # noqa: F401
from .genome import StrategyGenome
from .governor import RiskGovernor
from .lab import Lab, WARMUP
from .memory import Memory
from .novelty import state_series

log = logging.getLogger("rufflux.runner")


class HaltStore:
    """Persists the governor's latched HALT between runs (a JSON file in the repo)."""

    def __init__(self, path: str):
        self.path = path

    def load_into(self, gov: RiskGovernor) -> None:
        if os.path.exists(self.path):
            with open(self.path) as f:
                gov.restore_halt(json.load(f).get("reasons", []))

    def save(self, gov: RiskGovernor) -> None:
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"halted": gov.halted, "reasons": gov.halt_reasons}, f, indent=2)


class ResumableLab(Lab):
    """A Lab whose population, lineage and memory live in the DB, so each run advances ONE generation."""

    def __init__(self, cfg: LabConfig, bars: dict, mem: Memory):
        super().__init__(cfg, bars, mem, validate_span=False)
        self.manage_hist = False
        self._init_holdout()
        self._restore()

    def _init_holdout(self):
        dates = self.bars["dates"]
        if self.mem.get_meta("holdout_start") is None:   # sealed once, on the first ever run
            n = len(dates)
            self.mem.set_meta("holdout_start", dates[n - self.cfg.holdout_bars])
            self.mem.set_meta("holdout_end", dates[n - 1])
        h0 = int(np.searchsorted(dates, self.mem.get_meta("holdout_start"), side="left"))
        h1 = int(np.searchsorted(dates, self.mem.get_meta("holdout_end"), side="right"))
        self.holdout_range, self.n_dev = (h0, h1), h0
        if h0 < self.cfg.window + WARMUP:
            raise ValueError(f"not enough pre-holdout history ({h0} bars) for window {self.cfg.window}")

    def _restore(self):
        m = self.mem
        self.seen = {r["fingerprint"] for r in m.db.execute("SELECT fingerprint FROM strategies")}
        for r in m.db.execute("SELECT strategy_id, generation FROM experiments WHERE win=1"):
            self.wins.setdefault(r["strategy_id"], []).append(r["generation"])
        for r in m.db.execute("SELECT strategy_id, lesson FROM experiments ORDER BY id"):
            self.lessons[r["strategy_id"]] = (r["lesson"] or "").split(":")[0]
        self.pop = []
        for r in m.db.execute("SELECT id, genome FROM strategies WHERE status='alive' ORDER BY id"):
            h = m.db.execute("SELECT text FROM hypotheses WHERE strategy_id=? ORDER BY id DESC LIMIT 1", (r["id"],)).fetchone()
            self.pop.append({"sid": r["id"], "genome": StrategyGenome.from_dict(json.loads(r["genome"])),
                             "hypothesis": h["text"] if h else "restored"})
        last = m.db.execute("SELECT detail FROM events WHERE kind='generation_summary' ORDER BY id DESC LIMIT 1").fetchone()
        if last:
            self.survival.budget = float(json.loads(last["detail"]).get("risk_budget", 1.0))
        row = m.db.execute("SELECT MAX(generation) g FROM experiments").fetchone()
        self.next_gen = 0 if row["g"] is None else int(row["g"]) + 1

    def choose_window(self, gen: int) -> tuple:
        """Never overlaps the sealed holdout. Uses the latest post-holdout bars once there are
        enough of them; until then slides through pre-holdout history."""
        n, (h0, h1), w = len(self.bars["close"]), self.holdout_range, self.cfg.window
        if n - h1 >= w:
            return n - w, n
        span = h0 - w
        a = (gen * self.cfg.step) % (span + 1)
        return a, a + w

    def run_one(self, force: bool = False) -> dict | None:
        gen = self.next_gen
        a, b = self.choose_window(gen)
        h1 = self.holdout_range[1]
        latest = str(self.bars["dates"][-1])
        if a >= h1 and self.mem.get_meta("last_bar_date") == latest and not force:
            log.info("no new bars since %s; skipping generation", latest)
            return None
        self.rng = random.Random(self.cfg.seed * 100003 + gen)
        segs = [(0, min(a, self.holdout_range[0]))] + ([(h1, a)] if a > h1 else [])
        self.hist_states = []
        for s, e in segs:
            if e - s > WARMUP + 20:
                self.hist_states.extend(state_series(slice_bars(self.bars, s, e), WARMUP, every=10))
        summary = self.run_generation(gen, (a, b))
        self.mem.set_meta("last_bar_date", latest)
        self.next_gen = gen + 1
        summary["window"] = [str(self.bars["dates"][a]), str(self.bars["dates"][b - 1])]
        return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RUFFLUX daily runner (simulation only)")
    ap.add_argument("--symbols", default="SPY,QQQ")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--state-dir", default="state")
    ap.add_argument("--out", default="docs")
    ap.add_argument("--start", default="2016-01-01")
    ap.add_argument("--force", action="store_true", help="run a generation even without new bars")
    ap.add_argument("--reset-halt", action="store_true", help="human operator clears a latched HALT")
    ap.add_argument("--offline", action="store_true", help="use cached bars only")
    ap.add_argument("--today", default=None, help="override today's date (testing)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    from .dashboard import write_dashboard, write_index
    gov = RiskGovernor()
    store = HaltStore(os.path.join(a.state_dir, "halt.json"))
    store.load_into(gov)
    if a.reset_halt and gov.halted:
        log.warning("operator reset of HALT (was: %s)", gov.halt_reasons)
        gov.operator_reset("I-AM-THE-HUMAN-OPERATOR")
        store.save(gov)

    fetcher = AlpacaDataFetcher(a.data_dir)
    results, symbols = {}, [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    os.makedirs(a.state_dir, exist_ok=True)
    for sym in symbols:
        if gov.halted:
            log.error("HALTED (%s): no further runs", gov.halt_reasons)
            break
        try:
            if not a.offline:
                log.info("%s: +%d new bars", sym, fetcher.update(sym, start=a.start))
            bars = fetcher.load(sym)
        except Exception as e:  # network/credential problems are not a market-data-quality event
            log.error("%s: data fetch failed (%s); skipping this run", sym, e)
            results[sym] = {"error": str(e)}
            continue
        fatal, warns = check_quality(bars, today=a.today)
        for w in warns:
            log.warning("%s: %s", sym, w)
        if fatal:
            gov.data_quality_halt([f"{sym}:{f}" for f in fatal])
            store.save(gov)
            log.error("%s: DATA-QUALITY HALT %s", sym, fatal)
            break
        cfg = LabConfig(generations=10 ** 9, db_path=os.path.join(a.state_dir, f"{sym}.db"))
        mem = Memory(cfg.db_path)
        lab = ResumableLab(cfg, bars, mem)
        summary = lab.run_one(force=a.force)
        results[sym] = summary or {"skipped": "no new bars"}
        banner = f"{sym} · data through {bars['dates'][-1]} · generation {lab.next_gen}"
        write_dashboard(mem, cfg, RiskLimits(), os.path.join(a.out, f"{sym}.html"), banner=banner)
        mem.db.close()
    store.save(gov)
    write_index(results, gov, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
