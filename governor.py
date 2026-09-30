"""Risk Governor: an independent hard safety layer. Agents can REQUEST risk; only the governor
GRANTS it. Limits are a frozen dataclass copied at construction. A HALT latches and no AI
reasoning path can clear it (reset requires an explicit operator call outside the agents)."""
from __future__ import annotations
from dataclasses import dataclass, field
from .config import RiskLimits


@dataclass(frozen=True)
class RiskRequest:
    strategy_id: int
    notional: float          # proposed position notional
    gross_after: float = 0.0  # gross exposure after this position, as notional


@dataclass
class AccountState:
    capital: float
    peak_capital: float
    day_start_capital: float
    open_experiments: int = 0
    experimental_notional: float = 0.0
    gross_exposure: float = 0.0
    data_age_s: float = 0.0
    last_slippage_bps: float = 0.0


@dataclass(frozen=True)
class Decision:
    permitted_notional: float
    halted: bool
    reasons: tuple = ()


class RiskGovernor:
    def __init__(self, limits: RiskLimits | None = None):
        self._limits = limits or RiskLimits()
        self._halted = False
        self._halt_reasons: list = []

    @property
    def limits(self) -> RiskLimits:
        return self._limits

    @property
    def halted(self) -> bool:
        return self._halted

    def _halt(self, reason: str):
        self._halted = True
        self._halt_reasons.append(reason)

    def check_health(self, s: AccountState) -> list:
        """Evaluate hard boundaries. Any breach latches HALT."""
        L, breaches = self._limits, []
        if s.peak_capital > 0 and (s.peak_capital - s.capital) / s.peak_capital >= L.max_drawdown:
            breaches.append("max_drawdown")
        if s.day_start_capital > 0 and (s.day_start_capital - s.capital) / s.day_start_capital >= L.max_daily_loss:
            breaches.append("max_daily_loss")
        if s.data_age_s > L.max_data_staleness_s:
            breaches.append("data_quality_stale")
        if abs(s.last_slippage_bps) > L.max_slippage_anomaly_bps:
            breaches.append("execution_anomaly_slippage")
        for b in breaches:
            self._halt(b)
        return breaches

    def approve(self, req: RiskRequest, s: AccountState) -> Decision:
        """Return the notional the strategy is PERMITTED to risk (possibly 0)."""
        if self._halted:
            return Decision(0.0, True, ("halted:" + ",".join(self._halt_reasons),))
        breaches = self.check_health(s)
        if breaches:
            return Decision(0.0, True, tuple(breaches))
        L = self._limits
        if s.open_experiments >= L.max_simultaneous:
            return Decision(0.0, False, ("max_simultaneous",))
        room_pos = L.max_position_frac * s.capital
        room_exp = L.max_experimental_alloc * s.capital - s.experimental_notional
        room_gross = L.max_exposure * s.capital - s.gross_exposure
        permitted = max(0.0, min(req.notional, room_pos, room_exp, room_gross))
        reasons = () if permitted >= req.notional else ("clipped",)
        return Decision(permitted, False, reasons)

    @property
    def halt_reasons(self) -> list:
        return list(self._halt_reasons)

    def data_quality_halt(self, issues: list):
        """Data-quality shutdown: any fatal data issue latches HALT."""
        for i in issues:
            self._halt("data_quality:" + i)

    def restore_halt(self, reasons: list):
        """Re-latch a halt saved by a previous run (persistence is done by the runner, not by agents)."""
        for r in reasons:
            self._halt(r)

    def emergency_shutdown(self):
        self._halt("emergency_shutdown")

    def operator_reset(self, operator_confirmation: str):
        """Only for a human operator. Agents are never given a reference to this method."""
        if operator_confirmation != "I-AM-THE-HUMAN-OPERATOR":
            raise PermissionError("reset refused")
        self._halted = False
        self._halt_reasons.clear()
