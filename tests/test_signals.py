"""Tests for the lightweight signal bus."""

import pytest

from nexus.signals import Signal, SignalBus, SignalType


@pytest.mark.asyncio
async def test_signal_bus_emit_and_receive():
    """Handler should be called when matching signal is emitted."""
    bus = SignalBus()
    received = []

    async def handler(signal: Signal):
        received.append(signal)

    bus.on(SignalType.BREAKING_ALERT, handler)
    sig = Signal(type=SignalType.BREAKING_ALERT, payload={"topic": "tech", "count": 3})
    await bus.emit(sig)

    assert len(received) == 1
    assert received[0].type == SignalType.BREAKING_ALERT
    assert received[0].payload["topic"] == "tech"


@pytest.mark.asyncio
async def test_signal_bus_no_handler():
    """Emitting with no registered handlers should not raise."""
    bus = SignalBus()
    sig = Signal(type=SignalType.PIPELINE_COMPLETE, payload={})
    await bus.emit(sig)  # should not raise


@pytest.mark.asyncio
async def test_signal_bus_handler_error_isolated():
    """A failing handler should not prevent other handlers from running."""
    bus = SignalBus()
    results = []

    async def bad_handler(signal: Signal):
        raise RuntimeError("boom")

    async def good_handler(signal: Signal):
        results.append("ok")

    bus.on(SignalType.BREAKING_ALERT, bad_handler)
    bus.on(SignalType.BREAKING_ALERT, good_handler)

    sig = Signal(type=SignalType.BREAKING_ALERT)
    await bus.emit(sig)

    assert results == ["ok"]


@pytest.mark.asyncio
async def test_signal_bus_multiple_types():
    """Handlers should only fire for their registered signal type."""
    bus = SignalBus()
    breaking = []
    pipeline = []

    async def on_breaking(signal: Signal):
        breaking.append(signal)

    async def on_pipeline(signal: Signal):
        pipeline.append(signal)

    bus.on(SignalType.BREAKING_ALERT, on_breaking)
    bus.on(SignalType.PIPELINE_COMPLETE, on_pipeline)

    await bus.emit(Signal(type=SignalType.BREAKING_ALERT))
    await bus.emit(Signal(type=SignalType.PIPELINE_COMPLETE))
    await bus.emit(Signal(type=SignalType.BREAKING_ALERT))

    assert len(breaking) == 2
    assert len(pipeline) == 1


@pytest.mark.asyncio
async def test_signal_history_bounded():
    """Signal history should be capped to prevent unbounded growth."""
    bus = SignalBus()
    for i in range(150):
        await bus.emit(Signal(type=SignalType.PIPELINE_COMPLETE, payload={"i": i}))

    assert len(bus._history) <= 100


@pytest.mark.asyncio
async def test_signal_has_timestamp():
    """Signals should automatically get a timestamp."""
    sig = Signal(type=SignalType.BREAKING_ALERT)
    assert sig.timestamp is not None
