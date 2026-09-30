# RUFFLUX V0.1: evolutionary strategy lab (simulation only)

Run (no keys, no network needed):

    pip install -r requirements.txt
    python -m rufflux.lab --generations 10 --holdout     # writes rufflux.db + docs/index.html
    python -m unittest discover -s tests                 # 32 tests

## Layout
- genome.py / primitives.py: structured StrategyGenome + extensible primitive registry (no free-form code)
- data.py: seeded synthetic regime-switching market (swap in real bars via load_csv / Alpaca data later)
- sim.py: deterministic backtester (no lookahead, fees + slippage)
- evaluation.py: multi-objective Pareto selection, learning efficiency (kept OUTSIDE the generators)
- novelty.py: kNN novelty on the combined state, with a human-readable reason
- evolution.py / agents.py: mutation, crossover, simplification, radical exploration; rule-based Scientist
  (LLM plugs in via `genome_from_json`, which rejects anything that fails validation)
- memory.py: SQLite research memory: strategies, lineage, hypotheses, states, experiments, lessons
- governor.py: Risk Governor (frozen limits, latching HALT, operator-only reset)
- execution.py: ExecutionAgent seam. PaperAlpaca (paper endpoint only, governor-gated); LiveExecution always raises
- lab.py: the loop + SurvivalIntelligence + sealed holdout (every peek logged)
- dashboard.py: static HTML with RUFFLUXY (host on GitHub Pages / Vercel)

## Anti-overfitting in V0.1
Rolling walk-forward windows, out-of-sample validation slice used for selection, costs on every trade,
complexity penalty, and a sealed final holdout that only `holdout_report` can touch (logged).

## ## Unattended daily run (GitHub Actions)
1. Make an Alpaca account, generate PAPER keys, and add repo Secrets `ALPACA_KEY` and `ALPACA_SECRET`.
2. Settings > Pages > deploy from branch `main`, folder `/docs`.
3. Actions tab > "RUFFLUX daily" > Run workflow (first run pulls ~10 years of bars).
After that it runs weekdays at 22:30 UTC: tests -> fetch new bars -> data-quality gate -> one generation
per symbol (SPY, QQQ) -> dashboards -> commit `state/`, `data/`, `docs/`.

- The sealed holdout (last 500 bars at first run) is stored by date and never overlaps a generation window.
- Until ~600 new bars accumulate after the holdout, generations slide through pre-holdout history; after that
  they use the latest bars, and a run with no new bars is skipped.
- A data-quality failure (stale, gap > 7d, >50% one-day move, bad prices) latches HALT in `state/halt.json`.
  A network/credential failure only skips that run. To clear a HALT: run the workflow with `reset_halt` ticked.
- Free IEX feed = a slice of total volume. Prices are split-adjusted at fetch time; a later split would show up
  as an extreme-return jump in the cache and trip the halt (delete `data/<SYM>.csv` to refetch clean).

## Not in V0.1 (on purpose)
LLM Scientist, real market data, paper order flow, position monitor. Stages CONTROLLED/ESTABLISHED are unreachable.
