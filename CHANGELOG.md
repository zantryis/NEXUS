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
- 1,100+ tests (unit + integration + e2e)
