# Session State

## Current Phase
Phase 1 overhaul complete → backtest + evaluate → Phase 2

## What's Built
- Config: Pydantic models + YAML loader, per-topic filter thresholds + perspective diversity
- LLM: Multi-provider async client (Gemini, Anthropic, DeepSeek) with config_key routing
- Source polling: feedparser RSS, ContentItem model with affiliation/language metadata
- Source registries: 52 global feeds across 8 languages with affiliation tracking
- Ingestion: async (semaphore-limited), trafilatura extraction, language detection, paywall detection
- Filtering: two-pass (batch relevance → significance+novelty with knowledge context), perspective diversity
- Knowledge layer: Event extraction with entity tracking, structural dedup/merge, weekly compression
- Synthesis: TopicSynthesis (X) with NarrativeThread, convergence/divergence detection
- Renderers: text briefing from TopicSynthesis objects
- Evaluation: automated metrics (Shannon entropy, convergence ratio, language coverage), LLM-as-judge
- Pipeline orchestrator: poll → dedup → ingest → filter → extract → dedup → compress → synthesize → render
- Fixture capture/replay: FixtureCapture saves all intermediate data, FixtureReplay loads it
- Backtest infrastructure: partition by publish date, replay days chronologically, A/B model comparison
- CLI: `python -m nexus engine [--capture] [--backtest] [--label X] [--models-override key=value]`
- 109 passing tests (all mocked)

## Next Steps
1. Run capture + backtest: Gemini vs DeepSeek comparison
2. Evaluate synthesis quality, iterate on prompts
3. Phase 2: Audio pipeline (deferred)
