# References: Future Projection Research

Links, repositories, and sources organized by category.

---

## GraphRAG Ecosystem

### Microsoft GraphRAG
- **Repository**: https://github.com/microsoft/graphrag
- **Documentation**: https://microsoft.github.io/graphrag/
- Hierarchical community detection + LLM summarization for corpus-wide reasoning. Batch-oriented, poor incremental support. $20-500 per indexing run.

### Graphiti (by Zep)
- **Repository**: https://github.com/getzep/graphiti
- **Documentation**: https://docs.getzep.com/graphiti
- Real-time incremental knowledge graph updates, bi-temporal fact tracking, hybrid retrieval (semantic + BM25 + graph traversal). 94.8% on Deep Memory Retrieval. Requires Neo4j/FalkorDB/Kuzu.

### LazyGraphRAG (Microsoft)
- **Paper**: https://arxiv.org/abs/2410.20094
- 0.1% of full GraphRAG cost via deferred computation. Indexes deeply only when queries demand it.

### T-GRAG (Temporal GraphRAG)
- **Concept**: Bi-level temporal structure (knowledge graph + hierarchical time graph). 18x fewer prompt tokens for updates vs. full GraphRAG reindexing. 0.599 Correct vs. 0.410 baseline.

---

## MiroFish

### Repository & Project
- **Repository**: https://github.com/666ghj/MiroFish
- **Stars**: 16K+ (as of March 2026)
- 5-stage pipeline: GraphRAG → agent generation → dual-platform social simulation (OASIS/CAMEL-AI) → report synthesis → interactive analysis. Python 3.11 + Vue.js. No published prediction benchmarks.

### OASIS Framework (used by MiroFish)
- **Repository**: https://github.com/camel-ai/oasis
- Social simulation platform supporting up to 1M agents with 23 interaction types. Part of the CAMEL-AI ecosystem.

### Zep Cloud (used by MiroFish for agent memory)
- **Website**: https://www.getzep.com/
- Long-term memory for AI agents. Also the company behind Graphiti.

---

## Predictive Analytics from News

### Causal Knowledge Graphs from News
- **Approach**: BERT-based + pattern-based causal relation extraction, argument clustering via topic modeling. Transforms disconnected subgraphs into unified causal networks.
- **Relevance**: Formalizes Nexus's `relation_to_prior` text field into structured causal edges.

### GraphRAG-Causal
- **Performance**: 82.1% F1 with 20 few-shot examples, ~90% with 50. Hybrid semantic + structural scoring for causal reasoning in news.

### CAMEF (Causal-Aware Mixed-Frequency)
- **Approach**: Integrates textual + time-series data with causal learning and LLM-based counterfactual event augmentation for financial forecasting.

### Spatial-Temporal Knowledge Graph Networks
- **Approach**: Adds regional context to temporal KGs for geopolitical event prediction. Models trans-regional influence cascades.
- **Relevance**: Nexus tracks `country` on entities and sources — geographic correlation is available but unused.

---

## Agent Frameworks

### LangGraph
- **Repository**: https://github.com/langchain-ai/langgraph
- **Documentation**: https://langchain-ai.github.io/langgraph/
- Graph-based state machines, durable execution, human-in-the-loop, time-travel debugging. 47M+ PyPI downloads. Production leader for agent orchestration.

### CrewAI
- **Repository**: https://github.com/crewAIInc/crewAI
- **Documentation**: https://docs.crewai.com/
- Role-based multi-agent teams with intuitive delegation. Fastest time-to-value for structured multi-agent workflows.

### AutoGen (Microsoft)
- **Repository**: https://github.com/microsoft/autogen
- Now in maintenance mode (March 2026). Conversation-driven agent patterns. Not recommended for new projects.

### OpenAgents
- Persistent agent networks with native MCP + A2A protocols. Youngest framework in the space.

---

## Market Intelligence Tools

### Institutional Tier ($10K+/month)

| Tool | URL | Key Capability |
|---|---|---|
| Dataminr | https://www.dataminr.com/ | 50+ proprietary LLMs, 500K daily events, real-time risk scoring |
| Permutable AI | https://permutable.ai/ | Supply chain risk, ESG monitoring, NLP-powered intelligence |
| Recorded Future | https://www.recordedfuture.com/ | Threat intelligence, geopolitical risk, dark web monitoring |
| Acuity Trading | https://acuitytrading.com/ | Financial news NLP, sentiment, entity extraction |

### Consumer Tier

| Tool | URL | Price | Key Capability |
|---|---|---|---|
| TradeEasy.ai | https://tradeeasy.ai/ | Free | Shallow sentiment analysis |
| Trade Ideas | https://www.trade-ideas.com/ | $178-254/mo | AI trading signals (black-box) |
| Tickeron | https://tickeron.com/ | $125-250/mo | Pattern recognition, AI trading |
| TrendSpider | https://trendspider.com/ | $54-82/mo | Technical analysis automation |

### Adjacent Products

| Tool | URL | Relevance |
|---|---|---|
| Feedly Market Intelligence | https://feedly.com/market-intelligence | Closest comp — monitoring + AI, but no synthesis/memory |
| Perplexity | https://www.perplexity.ai/ | Research + Spaces, but oriented to internal knowledge |
| NotebookLM | https://notebooklm.google/ | Source-grounded research, but user-provided sources only |

---

## Calibration & Forecasting Methodology

### Superforecasting (Philip Tetlock)
- **Book**: "Superforecasting: The Art and Science of Prediction" (2015)
- Key concepts: calibration (stated confidence should match actual accuracy), Brier scores, reference class forecasting, updating beliefs incrementally.
- **Relevance**: Projection calibration loop (Tier 3b) should aim for Brier score tracking and reference class comparison.

### Metaculus
- **Website**: https://www.metaculus.com/
- Community forecasting platform with calibration benchmarks. Demonstrates that structured prediction with feedback loops can achieve meaningful accuracy.
- **Relevance**: Calibration methodology and question framing patterns applicable to projection design.

### Good Judgment Project
- **Website**: https://goodjudgment.com/
- Professional forecasting service using trained superforecasters. Demonstrates that structured analytical techniques improve prediction accuracy over unstructured intuition.

---

## Nexus Internal References

| Document | Path | Relevance |
|---|---|---|
| Product North Star | `docs/product-north-star.md` | Phase 3 "intelligence workspace for decisions" defines the product vision this feature serves |
| Pipeline Parameters | `docs/pipeline-parameters.md` | Thresholds and scoring rubrics that projection features must respect |
| Knowledge Store Schema | `src/nexus/engine/knowledge/schema.py` | 12-table schema that projection tables extend |
| Synthesis Diffing | `src/nexus/engine/synthesis/diff.py` | Existing "what changed" computation — projection builds on this |
| Experiment Framework | `src/nexus/engine/evaluation/experiment.py` | Benchmark suites for validating projection quality |
