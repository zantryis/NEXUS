# Future Projection: External Ecosystem Research

Research conducted March 2026 to assess technologies, approaches, and market landscape relevant to adding forward-looking intelligence capabilities to Nexus.

---

## 1. MiroFish — Agent Swarm Prediction Engine

**Repository**: [github.com/666ghj/MiroFish](https://github.com/666ghj/MiroFish) (16K+ stars)
**Stack**: Python 3.11 + Vue.js, any OpenAI-compatible LLM
**Creator**: Single Chinese undergraduate, backed by investor Chen Tianqiao

### Architecture — 5-Stage Pipeline

1. **Knowledge Construction**: GraphRAG parses seed materials (news, policy drafts, financial signals) into a structured knowledge graph preserving relational structure — influence networks, supply chains, social dynamics.

2. **Agent Generation**: Creates thousands of AI agents with distinct personalities, long-term memory (via Zep Cloud), and behavioral rules grounded in the knowledge graph. Each agent has a unique persona derived from the input data.

3. **Dual-Platform Simulation**: Agents interact on two social media-like platforms simultaneously (Twitter-like + Reddit-like). Powered by **OASIS** (CAMEL-AI framework, supports up to 1M agents, 23 interaction types). The dual-platform design captures different discourse dynamics — short-form vs. threaded discussion.

4. **Report Synthesis**: A dedicated ReportAgent analyzes emergent patterns — opinion shifts, coalition formation, narrative convergence — to generate structured prediction reports.

5. **Interactive Analysis**: Users can chat with individual agents, query the ReportAgent, or inject what-if scenarios to explore alternative outcomes.

### Assessment for Nexus

**Strengths**:
- The concept of generating forward-looking analysis from a knowledge graph is sound and directly relevant
- Interactive what-if scenarios are compelling for the "intelligence workspace for decisions" vision
- The GraphRAG → structured knowledge → analysis pipeline mirrors Nexus's own flow

**Weaknesses**:
- **No published benchmarks** comparing predictions to actual outcomes — the team acknowledges this gap
- LLM agents exhibit **amplified herd behavior** vs. real humans — simulated consensus is unreliable
- **Cost-prohibitive at scale**: team recommends <40 simulation rounds; costs escalate rapidly
- Zep Cloud dependency for agent memory adds external service cost
- Architecturally incompatible with Nexus's lean, budget-conscious pipeline ($1/day default)

**Verdict**: Directionally interesting as inspiration — particularly the knowledge graph → projection concept. But the simulation approach itself (thousands of agents with personalities debating on fake social media) is unproven, expensive, and adds complexity without demonstrated accuracy. **Use as inspiration, do not integrate.**

---

## 2. GraphRAG — Knowledge Graph + Retrieval-Augmented Generation

### Microsoft GraphRAG

**Repository**: [github.com/microsoft/graphrag](https://github.com/microsoft/graphrag)

**How it works** (6 phases):
1. Segment text into TextUnits
2. Map documents to chunks
3. LLM-based entity/relationship extraction and summarization
4. **Hierarchical Leiden Algorithm** for community detection (recursive clustering into hierarchical groups)
5. Community summarization (LLM-generated reports per community)
6. Vector embedding for retrieval

**Query modes**:
- **Global**: Corpus-wide reasoning via community summaries — the key innovation over traditional RAG
- **Local**: Entity-focused retrieval
- **DRIFT**: Hybrid of global + local
- **Basic**: Vector fallback

**Performance**: ~80-90% accuracy on complex multi-hop questions vs. ~50-67% for traditional RAG.

**Cost**: $20-500 for full indexing vs. $2-5 for vector RAG.

**Critical limitation for Nexus**: Batch-oriented. Incremental updates are poorly supported — entity extraction caches well, but graph construction and community detection must recompute. Microsoft is building `graphrag.append` but it is not production-ready. **Bad fit for Nexus's daily pipeline that appends events continuously.**

### Graphiti (by Zep)

**Repository**: [github.com/getzep/graphiti](https://github.com/getzep/graphiti)

This is the most relevant GraphRAG implementation for Nexus because it was designed for evolving, temporal data.

**Key capabilities**:
- **Real-time incremental updates** without batch recomputation — new episodes (events) are processed individually, matching Nexus's daily append-only event flow
- **Bi-temporal tracking**: Facts have validity windows (`t_valid`, `t_invalid`); old facts are invalidated, not deleted. This directly maps to how Nexus tracks thread lifecycle (emerging → active → stale → resolved) and could enhance convergence tracking
- **Hybrid retrieval**: Semantic + BM25 + graph traversal, sub-200ms latency. Nexus currently only has entity co-occurrence queries and LIKE-based search
- **Contradiction resolution**: Automatic with preserved history — when new information contradicts old, the system records both with temporal markers. Nexus has divergence detection but no automated contradiction resolution across time
- **Graph backends**: Neo4j, FalkorDB, Kuzu

**Performance**: 94.8% on Deep Memory Retrieval, up to 18.5% accuracy improvement on LongMemEval with 90% latency reduction.

**Assessment for Nexus**: The strongest candidate for enhancing Nexus's knowledge representation. However, Graphiti requires a graph database (Neo4j etc.), and Nexus's 12-table SQLite store is deeply integrated across 16+ web routes, QA pipeline, and Telegram bot. **Recommendation: Port Graphiti's core algorithms (bi-temporal validity, contradiction detection) into SQLite queries rather than adding a graph database dependency.**

### T-GRAG (Temporal GraphRAG)

Academic extension adding bi-level temporal structure:
- **Knowledge graph level**: Entity/relationship graphs with timestamps
- **Hierarchical time graph**: Temporal clustering for period-aware retrieval

**Performance**: 0.599 Correct vs. 0.410 baseline. **18x fewer prompt tokens** for updates than full GraphRAG reindexing.

**Relevance**: Nexus already separates event creation time (`events.created_at`) from event occurrence time (`events.date`). T-GRAG's approach validates that explicit temporal structure in the graph improves retrieval accuracy.

### LazyGraphRAG

Microsoft's cost-optimized variant: **0.1% of full GraphRAG cost** while maintaining competitive accuracy. Uses deferred computation — only indexes deeply when queries demand it.

**Relevance**: Given Nexus's budget constraint ($1/day default), LazyGraphRAG's philosophy of minimal upfront computation is well-aligned. Could inform how projection features are designed to be cost-efficient.

---

## 3. Predictive Analytics from News Text

### Causal Knowledge Graphs

**Approach**: BERT-based + pattern-based causal relation extraction from news text, then argument clustering via topic modeling. Transforms disconnected causal subgraphs into unified causal networks.

**Relevance to Nexus**: Events already have a `relation_to_prior` field (natural language text describing how an event relates to prior events). This is unstructured. Causal KGs would formalize this into directed edges: `Event A --[causes]--> Event B`. The extraction techniques (BERT-based + pattern-based) could be adapted or approximated via LLM prompts.

### GraphRAG-Causal

Combines graph retrieval with LLM inference for causal reasoning in news.

**Performance**: 82.1% F1 with 20 few-shot examples, ~90% with 50 examples. Uses hybrid semantic + structural scoring.

**Relevance**: Nexus already prompts for causal connections during synthesis (the prompt says "threads sharing 3+ entities and describing the same causal chain MUST be merged"). This implicit causal reasoning could be extracted as structured data rather than remaining embedded in synthesis prompts.

### CAMEF (Causal-Aware Mixed-Frequency)

Integrates textual + time-series data with causal learning and LLM-based counterfactual event augmentation for financial forecasting.

**Key insight**: Combining qualitative data (text summaries) with quantitative data (significance scores over time) is more powerful than either alone. Nexus captures both — event summaries (qualitative) and significance scores, event counts, source diversity metrics (quantitative) — but currently uses them independently.

### Spatial-Temporal Knowledge Graph Networks

Adds regional context to temporal knowledge graphs for geopolitical event prediction. Models trans-regional influence cascades (e.g., 9/11 → Afghanistan conflict).

**Relevance**: Nexus already tracks `country` on both entities and event sources. This metadata is currently used only for source independence checking (validating convergence requires different affiliations or countries). Geographic correlation could enable detecting influence cascades across regions.

### Supply Chain Cascade Prediction

Seven specialized agents detecting disruptions from news, mapping to multi-tier supplier networks, evaluating Tier-2 through Tier-4 propagation effects using graph neural networks.

**Relevance**: Demonstrates the value of second-order effect prediction — not just "what happened" but "what will this cause downstream." This is exactly the gap between Nexus's current "what happened" reporting and the desired "what should you do" intelligence.

---

## 4. Multi-Agent Frameworks

### LangGraph (Production Leader)

**Key features**: Graph-based state machines, durable execution, human-in-the-loop, time-travel debugging. 47M+ PyPI downloads. Used by Klarna, Replit, Elastic.

**Architecture fit**: Could orchestrate a projection workflow — analyst-agent reads thread history, forecaster-agent generates projections, critic-agent calibrates confidence. Steep learning curve but most battle-tested.

### CrewAI (Fastest Growing for Multi-Agent)

**Key features**: Role-based teams with intuitive delegation. Fastest time-to-value. Task-oriented focus.

**Architecture fit**: Maps naturally to the analyst/forecaster/critic pattern. Simpler than LangGraph for the specific use case of structured scenario generation.

### AutoGen (Microsoft) — Declining

Now in maintenance mode (Microsoft shifted focus). Conversation-driven, diverse dialogue patterns. **Not recommended for new integration.**

### OpenAgents

Persistent agent networks with native MCP + A2A protocols. Best for long-lived evolving knowledge communities. Youngest framework.

### 2026 Trend: "Agentic Mesh"

Modular ecosystems where different frameworks handle different roles. Example: LangGraph brain orchestrating CrewAI teams calling domain-specific tools. This is directionally aligned with Nexus's architecture — the LLMClient already supports different models per task via `config_key`.

### Assessment for Nexus

Nexus currently has zero multi-agent patterns. Everything goes through `LLMClient.complete()` with different `config_key` values. For Tier 1 projection features (thread trajectory, causal extraction, cross-topic bridging), no agent framework is needed — these are analytics and prompt extensions. For Tier 3 (multi-agent scenario generation), the simplest approach is sequential LLM calls via the existing client. A full agent framework is only justified if the workflow becomes genuinely complex (branching, human-in-the-loop, retry logic).

---

## 5. Market Landscape — Actionable News Intelligence

### Institutional Tier ($10K+/month)

| Tool | Key Capability | Price |
|---|---|---|
| **Dataminr** | 50+ proprietary LLMs, 500K daily events, real-time risk scoring | $10K+/mo minimum |
| **Permutable AI** | Supply chain risk, ESG monitoring, NLP-powered intelligence | Enterprise pricing |
| **Acuity NewsIQ** | Financial news NLP, sentiment, entity extraction | Enterprise pricing |
| **Recorded Future** | Threat intelligence, dark web monitoring, geopolitical risk | Enterprise pricing |

**What they offer that Nexus doesn't**: Real-time risk scoring, supply chain cascade mapping, calibrated confidence intervals, portfolio-specific impact assessment, compliance-ready reporting.

### Consumer Tier (Free–$250/month)

| Tool | Key Capability | Price |
|---|---|---|
| **TradeEasy.ai** | Shallow sentiment analysis | Free |
| **Trade Ideas** | Black-box trading signals | $178-254/mo |
| **Tickeron** | Pattern recognition, AI trading | $125-250/mo |
| **TrendSpider** | Technical analysis automation | $54-82/mo |

**What's missing at the $20-50/month price point**:
1. **Structured news impact mapping**: Not "bullish/bearish" but "tariff X affects Company Y through Supplier Z, expected impact in 2-6 weeks"
2. **Confidence-scored narratives**: WHY moves occur, connecting dots across sources
3. **Customizable alert thresholds**: User-defined, not vendor-dictated
4. **Portfolio-specific filtering**: Ignore irrelevant noise
5. **Causal chain visualization**: See the reasoning chain, not just the conclusion
6. **Temporal context**: How this compares to similar historical events
7. **Second-order effect prediction**: Knowledge graph-powered cascade analysis

### The Gap Nexus Can Fill

**Solo founders** now represent 36.3% of new startups (up from 23.7% in 2019). The need for institutional-quality intelligence at individual-accessible pricing is growing.

What makes a tool **actionable** vs. merely **informational**:
- **Specificity**: Exact company/sector/timeline, not vague sentiment
- **Timing**: When will impact materialize
- **Honest confidence calibration**: Acknowledge uncertainty
- **Decision frameworks**: "If you hold X, consider Y"
- **Historical pattern matching**: "This resembles the 2024 semiconductor situation"
- **Continuous knowledge accumulation**: Don't start fresh daily (Nexus's core strength)
- **Transparent contradiction handling**: Show where sources disagree (Nexus already does this)

Nexus is uniquely positioned here because it already has persistent topic memory, entity tracking, thread lifecycle, and convergence/divergence detection. Adding projection capabilities would create a product in the gap between consumer sentiment tools and enterprise intelligence platforms.
