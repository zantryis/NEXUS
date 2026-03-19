# Changelog

## [0.1.0] - 2026-03-18

### Added
- Multi-source news pipeline: RSS polling, content extraction, LLM filtering, event extraction
- SQLite knowledge store with entity resolution, narrative threads, and relationship graph
- TopicSynthesis engine with convergence/divergence analysis
- Audio pipeline: dialogue script generation + TTS (Gemini, ElevenLabs, OpenAI)
- Telegram bot: briefing delivery, Q&A, breaking news alerts, feedback
- FastAPI web dashboard: 16 routes (topics, threads, events, entities, predictions, podcast, settings, etc.)
- Multi-provider LLM client (Gemini, Anthropic, OpenAI, DeepSeek, Ollama, LiteLLM)
- Circuit breaker + retry logic with budget guard
- 6 prediction engines (actor, structural, graphrag, perspective, debate, naked baseline)
- Kalshi market integration: scanning, matching, forecasting, resolution, benchmarking
- Breaking news detection with wire feed polling + LLM scoring
- Setup wizard for first-run configuration
- Docker support (dev, test, production compose files)
- Feed health monitoring for source reliability tracking
- Breaking news feedback system (user reactions on alerts)
- Pipeline observability: per-stage timing, token usage, topic-level metrics
- Hindcast backtesting framework for prediction calibration
- Fixture capture/replay system for deterministic offline testing
- GitHub Pages showcase: interactive pipeline explorer + system overview
- 1,457 tests (unit + integration + e2e)
