"""Tests for BudgetConfig model."""

from nexus.config.models import BudgetConfig, NexusConfig, UserConfig


def test_budget_config_defaults():
    budget = BudgetConfig()
    assert budget.daily_limit_usd == 1.00
    assert budget.warning_threshold_usd == 0.50
    assert budget.degradation_strategy == "skip_expensive"


def test_budget_config_custom():
    budget = BudgetConfig(
        daily_limit_usd=5.00,
        warning_threshold_usd=3.00,
        degradation_strategy="stop_all",
    )
    assert budget.daily_limit_usd == 5.00
    assert budget.warning_threshold_usd == 3.00
    assert budget.degradation_strategy == "stop_all"


def test_budget_in_nexus_config():
    config = NexusConfig(
        user=UserConfig(name="Tristan"),
        budget=BudgetConfig(daily_limit_usd=2.00),
    )
    assert config.budget.daily_limit_usd == 2.00
    assert config.budget.warning_threshold_usd == 0.50


def test_budget_config_nexus_default():
    """NexusConfig gets a default BudgetConfig when none provided."""
    config = NexusConfig(user=UserConfig(name="Tristan"))
    assert config.budget.daily_limit_usd == 1.00
    assert config.budget.degradation_strategy == "skip_expensive"


def test_budget_config_in_yaml():
    """Load BudgetConfig from a dict, as YAML loader would produce."""
    data = {
        "user": {"name": "Tristan"},
        "budget": {
            "daily_limit_usd": 3.50,
            "warning_threshold_usd": 2.00,
            "degradation_strategy": "stop_all",
        },
    }
    config = NexusConfig(**data)
    assert config.budget.daily_limit_usd == 3.50
    assert config.budget.warning_threshold_usd == 2.00
    assert config.budget.degradation_strategy == "stop_all"
