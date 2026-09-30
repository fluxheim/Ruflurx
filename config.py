"""Hard limits and lab settings. Limits are frozen: agents get no write path to them."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimits:
    max_position_frac: float = 0.10       # of capital, per position
    max_experimental_alloc: float = 0.25  # of capital across all experiments
    max_daily_loss: float = 0.02          # fraction of capital
    max_drawdown: float = 0.10            # fraction of peak capital
    max_simultaneous: int = 5
    max_exposure: float = 0.50            # gross exposure / capital
    max_data_staleness_s: float = 900.0
    max_slippage_anomaly_bps: float = 50.0


@dataclass(frozen=True)
class LabConfig:
    seed: int = 7
    population: int = 24
    survivors: int = 8
    generations: int = 10
    window: int = 600          # bars visible to a generation
    step: int = 60             # bars the window advances per generation
    holdout_bars: int = 500    # sealed tail, never used by the loop
    fee_bps: float = 1.0
    slippage_bps: float = 2.0
    n_folds: int = 3
    complexity_weight: float = 0.02
    db_path: str = "rufflux.db"
    live_trading_enabled: bool = False  # V0.1: deliberately no code path honours True
