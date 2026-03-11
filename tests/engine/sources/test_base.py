"""Tests for source adapter base abstraction."""

import pytest

from nexus.engine.sources.base import SourceAdapter
from nexus.engine.sources.rss import RSSAdapter


def test_source_adapter_is_abstract():
    """SourceAdapter cannot be instantiated directly."""
    with pytest.raises(TypeError):
        SourceAdapter()


def test_rss_adapter_has_correct_type():
    adapter = RSSAdapter()
    assert adapter.source_type == "rss"
