# Nexus — System Specification

## What Is Nexus?

Nexus is a personal intelligence briefing system. It monitors news sources around the world, reads and analyzes articles using AI, identifies what matters, tracks how stories evolve over time, and delivers a daily briefing — as text and as a two-host podcast — straight to your phone.

Think of it as having a private analyst team that reads dozens of publications in eight languages, cross-references what different outlets are saying, spots emerging patterns, and gives you a concise morning report.

You pick the topics. Nexus handles the rest.

---

## What Does It Actually Do?

### 1. It Reads the News for You

Nexus monitors over 50 RSS feeds from news outlets, wire services, academic publishers, and government sources across eight languages. These include state media, public broadcasters, private outlets, nonprofits, and academic journals — deliberately chosen for editorial diversity.

Every few hours (and once comprehensively each morning), Nexus checks all feeds for new articles. It fetches the full text of each article, even when feeds only provide headlines or summaries.

### 2. It Decides What Matters

Not every article is worth your time. Nexus runs a two-stage filter:

- **Stage one** asks: "Is this article relevant to the topics I care about?" Each article gets a relevance score from 1 to 10. Low-scoring articles are dropped immediately.

- **Stage two** asks: "Is this significant? Is it telling me something new?" This second pass compares each article against what Nexus already knows from previous days. An article about the same meeting covered yesterday with no new information gets filtered out. A new development — even if minor — gets through.

This two-stage approach keeps signal high and noise low, while saving on AI processing costs by filtering out obvious misses early.

### 3. It Extracts Structured Facts

For every article that passes filtering, Nexus extracts a structured event: what happened, when, who was involved, what sources reported it, and how significant it is. These aren't summaries — they're structured records that can be searched, compared, and linked together.

### 4. It Builds a Knowledge Graph

Nexus doesn't just collect facts — it connects them.

**Entity resolution:** When one article mentions "President Trump," another says "Donald Trump," and a third says "Trump," Nexus recognizes these all refer to the same person. It maintains a canonical list of people, organizations, countries, treaties, and concepts, and links every event to the entities involved.

**Narrative threads:** Events that share entities and context are grouped into persistent story threads. A thread might be "US-Iran nuclear negotiations" or "EU AI regulation progress." Threads have lifecycles — they emerge, become active as more events accumulate, go stale when nothing new happens, and eventually resolve.

**Convergence and divergence detection:** When multiple independent sources confirm the same fact, Nexus flags it as convergence — you can trust this more. When different outlets report the same event but frame it differently (say, one calls it a "concession" and another calls it a "strategic repositioning"), Nexus flags that as divergence — the framing matters and you should know about it.

### 5. It Writes Your Briefing

Each morning, Nexus takes everything it has learned and generates a written briefing in markdown. The briefing includes:

- A two-to-three sentence executive summary
- A section per topic with the most important developments
- Key claims attributed to their sources (so you know who said what)
- A source tally showing the balance of perspectives

The briefing is concise — under 800 words — because the goal is to inform, not overwhelm.

### 6. It Records a Podcast

After writing the briefing, Nexus generates a natural two-host dialogue script. Two AI hosts (named Nova and Atlas) discuss the day's developments in a conversational format — not reading a script, but having an informed discussion that highlights what matters and why.

The script is then converted to audio using text-to-speech (with a choice of providers: Google's Gemini, OpenAI, or ElevenLabs). The audio segments are stitched together into a single podcast episode, saved as an MP3, and made available through a standard podcast RSS feed that works with any podcast app.

### 7. It Delivers to Your Phone

If you connect a Telegram bot, Nexus sends the briefing and podcast to your phone automatically each morning at a time you choose. It also supports:

- **Breaking news alerts:** Between daily briefings, Nexus periodically checks wire feeds for high-significance headlines. If something crosses your threshold (which you set), it sends an immediate alert.

- **Ask it questions:** You can text the bot a question about any topic and it will answer using its accumulated knowledge — recent events, thread context, entity relationships, convergence/divergence data. If its knowledge is thin on a subject, it falls back to web search.

- **Feedback:** After each briefing, you can rate it (thumbs up or down). Nexus stores this for quality tracking.

### 8. It Has a Dashboard

A web-based dashboard lets you explore everything Nexus knows:

- **Topics:** See all tracked topics with event counts, active threads, and filter statistics.
- **Threads:** Browse narrative threads by status (emerging, active, stale, resolved) and topic. Each thread page shows its events, what sources agree on (convergence), and where they disagree (divergence).
- **Events:** Browse all extracted events with sources, entities, and significance scores.
- **Entities:** Search the knowledge graph. Each entity has a profile page showing related events, threads, and co-occurring entities.
- **Sources:** See the balance of perspectives — how many events came from state media vs. private outlets vs. public broadcasters.
- **Filter log:** Full transparency into what got filtered out and why — every article Nexus saw is logged with scores and reasons.
- **Cost tracking:** See how much you're spending on AI calls per day.
- **Predictions:** Browse quantified forecasts with engine outputs, calibration data, and confidence levels.
- **Benchmark:** See how Nexus forecasts compare against Kalshi market prices, with Brier scores.
- **Changes:** View recent pipeline changes — what was extracted, filtered, and synthesized in each run.

### 9. It Makes Predictions

After building the knowledge graph, Nexus can generate calibrated probabilistic forecasts about how tracked stories will develop.

Six competing forecast engines approach each question from different angles:

- **Actor engine** (default) — Identifies relevant actors from the knowledge graph, reasons about each one's likely behavior, then synthesizes a probability
- **Structural engine** — Three-stage analysis inspired by academic forecasting research: base-rate reasoning, contrarian challenge, supervisor reconciliation
- **GraphRAG engine** — Walks the knowledge graph to find indirect connections the other engines miss
- **Debate engine** — Multiple AI personas reason independently, then debate and revise their estimates
- **Perspective engine** — Three analyst personas (evidence-based, contrarian, historical pattern) reason in parallel
- **Naked engine** — Baseline control: just the question and today's date, no context. If other engines can't beat this, the knowledge integration is adding noise, not signal.

Raw probabilities are calibrated to correct for LLM overconfidence: a compression function pushes extreme estimates toward 0.5. When a prediction market price is available, the system blends the LLM's estimate with the market anchor rather than replacing it.

Optionally, Nexus integrates with Kalshi prediction markets — scanning their settled markets, generating independent probabilities for the same questions, and comparing performance using Brier scores. A hindcast framework tests engines against historical data with strict date cutoffs so they can't peek at the answer.

---

## How Is It Configured?

### Topics

You define what you care about. Each topic has:

- **Subtopics** for precision (e.g., the topic "AI/ML Research" might have subtopics "LLM," "computer vision," "AI safety")
- **Source languages** — which languages to monitor
- **Perspective diversity** — how aggressively to enforce editorial balance (low, medium, high)
- **Filter threshold** — how strict the relevance filter should be (1-10 scale)
- **Scope** — narrow, medium, or broad (affects how many events are extracted per day)

Four topics ship with pre-built source registries: Iran-US Relations, AI/ML Research, Formula 1, and Global Energy Transition. For any other topic, Nexus can automatically discover relevant RSS feeds using AI.

### AI Providers

Nexus works with six AI providers:

| Provider | What It Offers | Cost |
|----------|---------------|------|
| **Ollama** (local) | Free, runs on your computer | $0/day |
| **DeepSeek** | Very cheap cloud AI | ~$0.01/day |
| **Google Gemini** | Good balance of cost and quality | ~$0.05/day |
| **OpenAI** (GPT-4) | High quality | ~$0.03–0.10/day |
| **Anthropic** (Claude) | High quality | ~$0.10/day |
| **LiteLLM** (cloud proxy) | Route to any model via OpenAI-compatible proxy | Varies |

You pick a preset (or mix and match per pipeline stage). Different parts of the system use different models — cheap, fast models for filtering and scoring; smarter, slower models for synthesis and dialogue writing.

### Budget Controls

You set a daily spending limit in dollars. When the limit approaches, Nexus either degrades to cheaper models or stops making AI calls entirely (your choice). It sends a warning via Telegram when spending crosses a threshold you define.

---

## How Does It Run?

Nexus runs as a single process on your computer or a server. That one process manages three things simultaneously:

1. **The web dashboard** — accessible at `localhost:8080`
2. **The scheduler** — triggers the daily pipeline and breaking news checks automatically
3. **The Telegram bot** — listens for commands and questions

All data lives in a single SQLite database file. No external databases, no cloud infrastructure beyond the AI provider APIs. You can back up the entire system by copying one folder.

It also runs in Docker if you prefer containerized deployment.

---

## What Makes This Different?

Most news aggregators give you a feed of articles. Nexus gives you **analyzed intelligence**:

- It doesn't just collect — it **filters** (two-pass, context-aware)
- It doesn't just summarize — it **extracts structured facts** and builds a knowledge graph
- It doesn't just list sources — it **detects when sources agree or disagree** on the same event
- It doesn't just show today — it **tracks narrative threads** across days and weeks
- It doesn't just read English — it **monitors eight languages** and normalizes everything into your language
- It doesn't just show you text — it **generates a podcast** with natural two-host dialogue
- It's fully transparent — every filtering decision is logged, every claim is attributed to its source

The system is self-hosted, open source, and designed for a single user who wants to stay deeply informed on specific topics without spending hours reading.
