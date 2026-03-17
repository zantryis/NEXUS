"""Tests for unified engine registry and config validation."""

import pytest

from nexus.engine.projection.forecasting import get_forecast_engine


def test_get_forecast_engine_actor():
    engine = get_forecast_engine("actor")
    from nexus.engine.projection.actor_engine import ActorForecastEngine
    assert isinstance(engine, ActorForecastEngine)


def test_get_forecast_engine_graphrag():
    engine = get_forecast_engine("graphrag")
    from nexus.engine.projection.graphrag_engine import GraphRAGForecastEngine
    assert isinstance(engine, GraphRAGForecastEngine)


def test_get_forecast_engine_perspective():
    engine = get_forecast_engine("perspective")
    from nexus.engine.projection.perspective_engine import PerspectiveForecastEngine
    assert isinstance(engine, PerspectiveForecastEngine)


def test_get_forecast_engine_debate():
    engine = get_forecast_engine("debate")
    from nexus.engine.projection.debate_engine import DebateBenchmarkEngine
    assert isinstance(engine, DebateBenchmarkEngine)


def test_get_forecast_engine_naked():
    engine = get_forecast_engine("naked")
    from nexus.engine.projection.naked_engine import NakedForecastEngine
    assert isinstance(engine, NakedForecastEngine)


def test_get_forecast_engine_unknown_raises():
    with pytest.raises(ValueError, match="Unknown forecast engine"):
        get_forecast_engine("nonexistent")


def test_config_accepts_all_engine_names():
    from nexus.config.models import FutureProjectionConfig

    for name in ("actor", "native", "graphrag", "perspective", "debate", "naked", "structural"):
        cfg = FutureProjectionConfig(engine=name)
        assert cfg.engine == name
