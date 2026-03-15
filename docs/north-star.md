# Nexus: what it is, what it isn't, what it should become

## What this system actually does

Nexus is a self-hosted pipeline that turns noisy multi-source news feeds into structured intelligence with source attribution, conflict detection, and persistent memory. It runs daily, stays within budget, and delivers through dashboard, Telegram, and podcast.

The core loop:

1. Poll RSS feeds across configured topics
2. Ingest and extract full text
3. Two-pass filter: relevance scoring, then significance + novelty assessment with knowledge context
4. Extract structured events with source metadata
5. Deduplicate events (entity overlap + date proximity, no LLM call)
6. Resolve and canonicalize entities
7. Compress old events into weekly summaries
8. Synthesize into TopicSynthesis — the intermediate knowledge object with narrative threads, convergence, and divergence
9. Render artifacts: text briefing, podcast dialogue, dashboard views
10. Deliver via Telegram, dashboard, and audio

Everything downstream renders from TopicSynthesis. That's the architectural decision that matters most.

## What makes this different from the obvious comparisons

**It's not a news summarizer.** Summarizers take articles in, produce shorter text out. Nexus maintains a knowledge graph across runs — events, entities, threads, convergence, divergence. Today's briefing knows what happened yesterday. That's the difference between a function call and a system.

**It's not a general agent.** There's no free-form tool loop, no "do anything" promise. The pipeline is 13 explicit stages with known cost characteristics. You can predict what a run costs before it starts. You can debug a bad briefing by looking at the filter log, the event extraction, or the synthesis prompt. General agents can't offer that.

**It's not a feed reader.** Feedly is better at feed management and will always be. Nexus is better at synthesis and memory — turning what 40 sources said into 5 narrative threads with convergence and divergence analysis. The output is a knowledge object, not a reading list.

**It's not a research notebook.** NotebookLM starts from documents you bring. Nexus starts from continuous monitoring of the world. You define topics, it watches, it remembers. You don't have to curate inputs.

## What this system is for

Nexus is for people who need to stay continuously informed on 3-10 external topics and cannot afford to miss important developments.

Concretely:

- A founder tracking AI, regulation, and three competitors
- A policy analyst tracking sanctions, energy markets, and nuclear programs
- An investor monitoring sectors and geopolitical risk
- A researcher tracking a field across institutions, preprints, and industry moves

The common thread: recurring topics, high cost of missing something, daily/weekly consumption habit, preference for structured intelligence over raw feed volume.

## Design principles (from the code, not aspirations)

### 1. The pipeline is deterministic in structure

13 stages, always in the same order, with explicit handoffs between Pydantic models. This means you can:
- Debug by inspecting intermediate artifacts
- Measure cost per stage
- Swap models per stage without changing the pipeline
- Write tests against each stage independently (760+ tests exist)

This is not a constraint to overcome. It's the reason the system works reliably.

### 2. Cost control is a first-class concern

- BudgetGuard with daily limits, warning thresholds, and degradation strategies
- Two degradation modes: skip_expensive (blocks synthesis/dialogue only) or stop_all
- Budget syncs from SQLite across process restarts
- Fire-and-forget cost logging so DB writes don't block LLM calls
- Per-provider cost estimation
- Model presets from free (Ollama local) to quality (Gemini Pro), with realistic daily cost estimates

This matters because a system that costs $5/day to run will be turned off. A system that costs $0.05/day gets used.

### 3. TopicSynthesis is the core product, not the briefing

Briefings, podcasts, Q&A answers, and dashboard views all render FROM the same TopicSynthesis object. This means:
- New delivery formats don't require pipeline changes
- Quality improvements to synthesis propagate to all outputs
- Testing synthesis quality tests all outputs implicitly

### 4. Source metadata flows through the entire pipeline

Every ContentItem carries affiliation (state/public/private), country, tier, and language. This metadata survives through filtering, event extraction, and synthesis. It enables:
- Perspective diversity enforcement in filtering
- Convergence analysis (are confirming sources actually independent?)
- Divergence detection (which outlets frame this differently?)
- Source balance reporting

### 5. Evaluation is built in, not bolted on

- Experiment framework with 8 suites (A-H) testing pipeline quality, threshold sensitivity, diversity impact, style comparison, cross-judge validation, weight sensitivity, model combinations, and divergence prompt variants
- LLM-as-judge with anchored rubric (completeness, source balance, convergence accuracy, divergence detection, entity coverage)
- Multi-judge validation (Gemini Pro + DeepSeek Reasoner + cloud judges)
- Fixture capture/replay for offline testing
- Cross-environment comparison via fixture export/import

## What should change

### Near-term (trust and observability)

**"What changed since last briefing"** is the single highest-value missing feature. The synthesis snapshots already exist in SQLite. The diff is a comparison function, not a pipeline change. This should be a first-class artifact — not buried in a dashboard, but delivered alongside the briefing.

**Filter observability** needs to be surfaced. The filter_log table captures every accept/reject decision with scores and reasons. Users should be able to see what got rejected and why, without querying SQLite directly. This is how you answer "am I over-filtering?" with data.

**Adjacent signal detection** — articles from existing sources that score high on significance but low on relevance to all configured topics. These are the "things you didn't know you should be tracking" signal. It's a query on existing data, not a new pipeline.

**Scope-dependent significance thresholds** — the pass-2 gate should vary by topic scope. A narrow topic with few sources can't afford the same rejection rate as a broad topic with 35+ articles. Narrow=3, medium=4, broad=5.

### Medium-term (memory and retention)

**Thread timelines** — threads are persisted with status lifecycle (emerging → active → stale → resolved) and linked events. The dashboard should show thread evolution over time, not just current state.

**Entity watchlists** — let users mark entities they want to track closely. When a watched entity appears in a new event, surface it prominently.

**Change detection confidence** — not just "what changed" but "how confident are we this is real?" The convergence/divergence machinery already answers this. Surface the confidence level per development.

**Source health monitoring** — which feeds are returning articles? Which are dead? Which produce content that consistently gets filtered out? This is data the pipeline already generates but doesn't surface.

### What not to build

**Don't add general-purpose document storage.** Academic papers, PDFs, notes — these are different knowledge types. The event/thread/convergence/divergence model is designed for temporal news narratives. Forcing other content types into it would weaken the model without serving any content type well.

**Don't add multi-user before single-user is trusted.** Multi-user means auth, permissions, shared state, conflict resolution. That's a rewrite of the web layer and probably a move off SQLite. The single-user experience needs to be something you trust every morning before any of that matters.

**Don't chase "agent" branding.** The system has agentic characteristics (Q&A with knowledge grounding, breaking news detection, source discovery). But positioning it as "an agent" sets expectations of general-purpose action-taking. The value is the structured intelligence pipeline, not the conversational interface.

**Don't add action-taking.** "Nexus noticed a development and automatically drafted an email to your team" sounds compelling in a demo. In practice, it means the system does things you didn't ask for with confidence levels you can't verify. The right boundary is: Nexus watches and tells you. You decide what to do.

## Metrics that matter

### Is it working?
- % of scheduled briefings delivered on time
- % of podcast runs that produce playable audio
- Pipeline completion rate
- Telegram response success rate

### Is it useful?
- Days per week the briefing is consumed
- Q&A queries against topic memory
- Number of active topics per user
- Number of delivery surfaces used (dashboard, Telegram, audio)

### Is it good?
- Synthesis quality scores (completeness, source balance, convergence accuracy)
- Filter pass rate by topic (too low = over-filtering, too high = no signal)
- Event novelty precision (are "novel" events actually new?)
- Adjacent signal discovery rate

## What this system could become

The best version of Nexus is not "AI that summarizes the news." It's software that keeps watch on the external world, builds memory over time, and helps you act with better context.

That's a real product for a real audience: people whose work depends on understanding what's happening outside their organization, across multiple domains, continuously.

The path from here to there is not adding features. It's:
1. Trusting the daily output enough to rely on it
2. Seeing what got filtered and why
3. Noticing when something changed that matters
4. Knowing when the system is confident vs uncertain

Everything else follows from those four capabilities being reliable.
