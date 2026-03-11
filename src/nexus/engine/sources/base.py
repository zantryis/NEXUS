"""Base source adapter abstraction."""

from abc import ABC, abstractmethod

from nexus.engine.sources.polling import ContentItem


class SourceAdapter(ABC):
    """Abstract base for all source adapters."""

    source_type: str

    @abstractmethod
    async def poll(self, source_config: dict) -> list[ContentItem]:
        """Fetch items from a single source config entry."""
        ...
