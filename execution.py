"""Execution interface. Paper-only in V0.1. LiveExecution exists so the seam is visible, and is
deliberately unusable. Every order must be cleared by the Risk Governor."""
from __future__ import annotations
import json
import os
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from .governor import RiskGovernor, RiskRequest, AccountState

PAPER_BASE_URL = "https://paper-api.alpaca.markets"


class LiveTradingDisabled(RuntimeError):
    pass


class ExecutionRefused(RuntimeError):
    pass


@dataclass(frozen=True)
class Order:
    strategy_id: int
    symbol: str
    side: str      # buy|sell
    notional: float


class ExecutionAgent(ABC):
    @abstractmethod
    def submit(self, order: Order, state: AccountState) -> dict: ...


class PaperAlpaca(ExecutionAgent):
    def __init__(self, governor: RiskGovernor, base_url: str = PAPER_BASE_URL, key=None, secret=None, transport=None):
        if governor is None:
            raise ExecutionRefused("a RiskGovernor is required")
        if base_url.rstrip("/") != PAPER_BASE_URL:
            raise ExecutionRefused("only the Alpaca paper endpoint is allowed in V0.1")
        self.governor = governor
        self.base_url = base_url
        self._key = key or os.environ.get("ALPACA_KEY")
        self._secret = secret or os.environ.get("ALPACA_SECRET")
        self._transport = transport or self._http

    def submit(self, order: Order, state: AccountState) -> dict:
        decision = self.governor.approve(RiskRequest(order.strategy_id, order.notional), state)
        if decision.halted:
            raise ExecutionRefused("HALT: " + ";".join(decision.reasons))
        if decision.permitted_notional <= 0:
            raise ExecutionRefused("governor permitted 0: " + ";".join(decision.reasons))
        if not (self._key and self._secret):
            raise ExecutionRefused("missing ALPACA_KEY / ALPACA_SECRET")
        body = {"symbol": order.symbol, "notional": round(decision.permitted_notional, 2),
                "side": order.side, "type": "market", "time_in_force": "day"}
        return self._transport("/v2/orders", body)

    def _http(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.base_url + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"APCA-API-KEY-ID": self._key, "APCA-API-SECRET-KEY": self._secret,
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())


class LiveExecution(ExecutionAgent):
    def __init__(self, *a, **k):
        raise LiveTradingDisabled("Live execution is disabled in RUFFLUX V0.1.")

    def submit(self, order, state):  # pragma: no cover
        raise LiveTradingDisabled("Live execution is disabled in RUFFLUX V0.1.")
