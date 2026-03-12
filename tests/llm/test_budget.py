"""Tests for BudgetGuard logic."""

import logging

import pytest
from nexus.config.models import BudgetConfig
from nexus.llm.budget import BudgetGuard


@pytest.fixture
def default_config():
    return BudgetConfig()


@pytest.fixture
def stop_all_config():
    return BudgetConfig(degradation_strategy="stop_all")


@pytest.fixture
def guard(default_config):
    return BudgetGuard(default_config)


def test_budget_guard_initial_spend_zero(guard):
    assert guard.today_spend == 0.0


def test_budget_guard_record_cost(guard):
    guard.record_cost(0.10)
    assert guard.today_spend == pytest.approx(0.10)
    guard.record_cost(0.05)
    assert guard.today_spend == pytest.approx(0.15)


def test_budget_check_ok(guard):
    """Under warning threshold returns 'ok'."""
    guard.record_cost(0.20)
    assert guard.check_budget("filtering") == "ok"


def test_budget_check_warning(guard):
    """Between warning threshold and limit returns 'warning'."""
    guard.record_cost(0.60)
    assert guard.check_budget("filtering") == "warning"


def test_budget_check_degraded(guard):
    """Over limit with skip_expensive strategy and expensive key returns 'degraded'."""
    guard.record_cost(1.50)
    assert guard.check_budget("synthesis") == "degraded"
    assert guard.check_budget("dialogue_script") == "degraded"
    assert guard.check_budget("agent") == "degraded"


def test_budget_check_blocked_stop_all(stop_all_config):
    """Over limit with stop_all strategy returns 'blocked'."""
    guard = BudgetGuard(stop_all_config)
    guard.record_cost(1.50)
    assert guard.check_budget("filtering") == "blocked"
    assert guard.check_budget("synthesis") == "blocked"


def test_budget_cheap_key_allowed_over_limit(guard):
    """Cheap keys still allowed when over limit in skip_expensive mode."""
    guard.record_cost(1.50)
    assert guard.check_budget("filtering") == "warning"
    assert guard.check_budget("breaking_news") == "warning"
    assert guard.check_budget("knowledge_summary") == "warning"


def test_budget_warning_sent_once(guard, caplog):
    """Warning is logged only once per day."""
    guard.record_cost(0.60)
    with caplog.at_level(logging.WARNING):
        result1 = guard.check_budget("filtering")
        result2 = guard.check_budget("filtering")
    assert result1 == "warning"
    assert result2 == "warning"
    # Warning should appear only once
    warning_msgs = [r for r in caplog.records if "Budget warning" in r.message]
    assert len(warning_msgs) == 1
