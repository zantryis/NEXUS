"""Budget enforcement for LLM API calls."""

import logging
from datetime import date

from nexus.config.models import BudgetConfig

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Daily budget limit reached, all LLM calls blocked."""
    pass


class BudgetDegradedError(Exception):
    """Daily budget limit reached for expensive operations."""
    pass


# Config keys considered expensive (use pro/slow models)
EXPENSIVE_KEYS = {"synthesis", "dialogue_script", "agent"}


class BudgetGuard:
    """Tracks daily spend and enforces budget limits."""

    def __init__(self, config: BudgetConfig):
        self._config = config
        self._daily_spend: dict[str, float] = {}  # date_str -> usd
        self._warning_sent: set[str] = set()  # dates we've already warned about

    @property
    def today_spend(self) -> float:
        return self._daily_spend.get(date.today().isoformat(), 0.0)

    def record_cost(self, cost_usd: float):
        today = date.today().isoformat()
        self._daily_spend[today] = self._daily_spend.get(today, 0.0) + cost_usd

    def check_budget(self, config_key: str) -> str:
        """Check budget status before an LLM call.

        Returns: "ok" | "warning" | "degraded" | "blocked"

        - "ok": under warning threshold
        - "warning": over warning threshold but under limit
        - "degraded": over limit, skip_expensive strategy, and this is an expensive config_key
        - "blocked": over limit, stop_all strategy (or skip_expensive and this is cheap too)
        """
        spend = self.today_spend
        limit = self._config.daily_limit_usd
        warning = self._config.warning_threshold_usd
        strategy = self._config.degradation_strategy

        if spend < warning:
            return "ok"

        if spend < limit:
            today = date.today().isoformat()
            if today not in self._warning_sent:
                self._warning_sent.add(today)
                logger.warning(f"Budget warning: ${spend:.4f} of ${limit:.2f} daily limit")
            return "warning"

        # Over limit
        if strategy == "stop_all":
            return "blocked"

        # skip_expensive: allow cheap ops, block expensive ones
        if config_key in EXPENSIVE_KEYS:
            return "degraded"

        return "warning"  # cheap ops still allowed in skip_expensive mode

    async def sync_from_store(self, store) -> None:
        """Load today's accumulated spend from the persistent store.

        Called on startup so budget enforcement survives process restarts.
        """
        today = date.today().isoformat()
        cost = await store.get_daily_cost(today)
        if cost > 0:
            self._daily_spend[today] = cost
            logger.info(f"Budget synced from store: ${cost:.4f} spent today")

    @property
    def should_warn(self) -> bool:
        today = date.today().isoformat()
        return (
            self.today_spend >= self._config.warning_threshold_usd
            and today not in self._warning_sent
        )
