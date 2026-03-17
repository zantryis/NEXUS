# Executive Assessment: Future Projection Capabilities for Nexus

## TL;DR

The idea is strongly viable. Nexus's persistent knowledge base — entities tracked across time, narrative threads with lifecycle states, cross-source convergence/divergence, temporal summaries — is an unusually strong foundation for projection. Most news intelligence tools start fresh every day; Nexus already has cumulative memory. The product north star already calls out Phase 3 as "intelligence workspace for decisions" with "scenario monitoring / signposts." This feature is on the roadmap. The question is not whether, but how and when.

---

## 1. Is the Idea Viable?

**Yes — strongly.**

Three factors make this more than speculative:

1. **The data foundation exists.** Nexus already tracks entities across topics, maintains narrative threads with lifecycle states, detects convergence/divergence across sources, and compresses knowledge into temporal summaries. Most of the raw material for projection is already being captured — it just isn't being analyzed for forward-looking signals.

2. **The market gap is real.** Institutional intelligence tools (Dataminr, Recorded Future) cost $10K+/month. Consumer tools offer shallow sentiment analysis. At the $20-50/month price point — exactly where solo founders, independent traders, and small-team analysts live — there is nothing that provides structured impact mapping, causal chains, or confidence-calibrated forward-looking assessments. Nexus can occupy this gap.

3. **The product vision already calls for it.** The north star doc (Phase 3) explicitly describes an "intelligence workspace for decisions" with "scenario monitoring / signposts" and "decision memos built from tracked topics." Projection capabilities are the concrete implementation of that vision.

---

## 2. MiroFish: Should We Build On It?

**No — use as inspiration only.**

MiroFish demonstrates that GraphRAG → knowledge graph → forward-looking analysis is a compelling concept. But the specific implementation — simulating thousands of AI agents with distinct personalities debating on fake social media platforms — has serious problems:

- **No published benchmarks** comparing predictions to actual outcomes
- **Amplified herd behavior**: LLM agents converge on consensus faster than real humans, producing false confidence
- **Cost-prohibitive**: Recommended <40 simulation rounds; costs escalate rapidly with agent count
- **Architectural mismatch**: MiroFish requires Zep Cloud, OASIS framework, and significant compute — incompatible with Nexus's $1/day budget and lean SQLite architecture

**What to take from MiroFish**: The concept of structured knowledge → forward-looking analysis is valid. The specific mechanism (agent simulation) is not the right tool for Nexus.

---

## 3. Graphiti: Should We Integrate It?

**Port concepts, don't integrate the system.**

Graphiti (by Zep) has the best ideas for evolving knowledge graphs:
- **Bi-temporal tracking**: Facts have validity windows (when a fact was believed true, when it was superseded). This is exactly what Nexus's convergence table needs.
- **Contradiction resolution**: Automatic detection and recording of when new information contradicts old information. Nexus has divergence detection but no temporal resolution tracking.
- **Incremental updates**: New episodes processed without rebuilding the graph. Matches Nexus's daily append-only flow.

**Why not integrate directly**: Graphiti requires Neo4j (or FalkorDB/Kuzu). Nexus's SQLite store is used by 16+ web routes, the QA pipeline, the Telegram bot, the experiment framework, and the migration system (9 successful schema versions). Replacing it would be a multi-week effort with high regression risk across the entire system.

**Better approach**: Implement Graphiti's key concepts as SQLite columns and queries:
- Add `valid_from` / `valid_until` / `superseded_by` to the convergence table
- Add a weekly job that detects when convergence facts have been contradicted
- Keep the existing store architecture intact

---

## 4. Recommended Approach: Phased Implementation

### Phase A (First Sprint) — Pure Analytics, Zero LLM Cost

**Implement**: Tier 1a (thread trajectory) + Tier 1c (cross-topic entity bridging)

**Why start here**:
- Thread trajectory requires one new table (`thread_snapshots`) and pure SQL/Python analytics — no LLM calls
- Cross-topic bridging is a single SQL query against the existing schema
- Both provide immediate, tangible value in briefings and dashboard
- Both are fully testable with existing fixtures
- Zero impact on the daily LLM budget

**Deliverables**: "This thread is accelerating" annotations on briefings. "Entity X is active across topics A and B" cross-topic alerts.

### Phase B (Second Sprint) — Causal + Forward Look

**Implement**: Tier 1b (causal relation extraction) + Tier 1d (projection artifact)

**Why second**:
- Causal extraction extends an existing LLM prompt (the `relation_to_prior` field) — minimal additional cost
- The projection artifact consumes trajectory + causal + cross-topic data from Phase A
- This is where users first see genuinely forward-looking content: "Based on thread trajectory and causal chain, here's what to watch for"

**Deliverables**: "Forward Look" section in daily briefings with confidence-rated assessments and signposts.

### Phase C (After 30+ Days of Data) — Temporal Depth

**Implement**: Tier 2a (temporal validity windows) + Tier 3b (calibration loop)

**Why gated on data depth**:
- Temporal validity windows need weeks of convergence records to detect supersession patterns
- Calibration requires comparing projections to actual outcomes — needs at least 30 days of projections from Phase B

**Deliverables**: "What was believed true on date X?" queries. Projection accuracy stats feeding back into prompt engineering.

### Phase D (User-Demand Driven) — Experimental

**Implement**: Tier 3a (multi-agent scenarios) + Tier 2b (source reliability scoring)

**Why last**:
- Multi-agent scenarios are the most speculative — value is uncertain without calibration data from Phase C
- Source reliability requires significant accumulated data (20+ claims per outlet per topic)
- Both should be driven by user feedback, not preemptive engineering

**Deliverables**: Structured scenario exploration ("if X, then Y is likely"). Source-weighted confidence in projections.

---

## 5. Risk Assessment

### Hallucination Risk — **HIGH concern, MEDIUM after mitigation**

LLMs generating forward-looking projections will sometimes be wrong. This is the most serious risk to product trust.

**Mitigations**:
1. Every projection must cite specific data (which thread trajectory, which causal chain, which cross-topic signal). No free-form speculation.
2. Confidence levels are mandatory: LOW (single signal), MEDIUM (2+ corroborating signals), HIGH (strong trajectory + causal + cross-topic confirmation).
3. Projections are framed as "based on current trajectory" not "will happen."
4. Signposts accompany every projection ("watch for X to confirm/disconfirm this").
5. Calibration loop (Phase C) provides empirical accuracy data.

### Complexity Creep — **MEDIUM concern**

The CLAUDE.md says "no bloat: minimal code, no over-engineering, no premature abstractions." Four tiers of projection capabilities could violate this principle.

**Mitigations**:
1. Tier gating — each phase must demonstrate user value before the next begins
2. Tier 1 is lean by design: one new table, two new modules, extensions to existing prompts
3. Tier 3 is explicitly optional and user-demand-driven
4. No new external dependencies in any tier

### Budget Impact — **LOW concern for Tier 1, MEDIUM for Tier 3**

| Phase | Additional LLM Calls per Run | Estimated Cost per Run |
|---|---|---|
| Phase A (trajectory + cross-topic) | 0 | $0.00 |
| Phase B (causal + projection) | 1-2 per topic | ~$0.005 |
| Phase C (calibration) | 1 per week | ~$0.001/day |
| Phase D (scenarios) | 6-8 per run | ~$0.01-0.08 |

At the `balanced` budget preset ($0.05/day), even Phase D is feasible. But it should be configurable (opt-in for projection features).

### Data Sparsity — **HIGH concern early, resolves over time**

A fresh Nexus install with 1 week of data cannot meaningfully project. Thread trajectory needs at least 3 daily snapshots. Causal chains need a history of events to link. Calibration needs 30+ days.

**Mitigations**:
1. Minimum data thresholds: trajectory requires 3+ snapshots, projection artifact requires 7+ days of events, calibration requires 30+ reviewed projections
2. Graceful degradation: if insufficient data, omit projection section from briefings (don't generate hollow projections)
3. Clear messaging: "Projection features will activate after X more days of data collection"

---

## 6. Build vs. Integrate Decision Matrix

| Capability | Build | Integrate | Recommendation | Rationale |
|---|---|---|---|---|
| Thread trajectory | Build | N/A | **Build** | Simple SQL analytics on existing data; no external tool does this for Nexus's specific schema |
| Causal extraction | Build | GraphRAG-Causal prompts | **Build** | Extend existing extraction prompt; importing a causal extraction library would add complexity for marginal accuracy gain |
| Cross-topic correlation | Build | N/A | **Build** | One SQL query against existing schema; no integration candidate exists |
| Temporal validity | Build (port Graphiti concepts) | Graphiti | **Build** | Porting algorithms into SQLite is safer than adding a graph database dependency |
| Source reliability | Build | N/A | **Build** | Straightforward accumulation; no off-the-shelf tool fits Nexus's source model |
| Scenario generation | Build | LangGraph orchestration | **Build initially** | 3 sequential LLM calls don't need an orchestration framework; reconsider if workflow becomes complex |
| Calibration loop | Build | N/A | **Build** | Simple table + comparison logic; unique to Nexus's projection output |

**Net**: Build everything. No external integration justified at this stage. The risk of adding dependencies (Neo4j, LangGraph, MiroFish's OASIS) exceeds the value for Nexus's lean, self-hosted architecture.

---

## 7. Success Metrics

### Phase A (immediate, no projections yet)
- Thread trajectory classification accuracy: does "accelerating" correlate with actual significance increase in 7 days? (backtest against existing synthesis snapshots)
- Cross-topic entity coverage: % of entities appearing in 2+ topics that are detected

### Phase B (after projection artifact ships)
- User engagement with "Forward Look" section: read rate, Q&A queries about projections
- Projection specificity: average number of data citations per projection (target: 2+)

### Phase C (after 60+ days)
- **Projection calibration**: % of projections directionally correct after review period
  - Target: LOW confidence = 30-40% correct, MEDIUM = 50-60%, HIGH = 70%+
  - If LOW > HIGH, the confidence model is miscalibrated
- Cost delta: actual daily LLM spend increase vs. pre-projection baseline

### Phase D (if implemented)
- Scenario coverage: % of significant events that were anticipated by at least one scenario
- Source reliability correlation: do higher-accuracy sources produce more reliable projections?

---

## 8. Alignment with Product North Star

| North Star Principle | Projection Feature Alignment |
|---|---|
| "Reliability beats breadth" | Tier gating ensures projection features don't compromise core briefing reliability |
| "Memory beats chat" | Projection is built on accumulated knowledge — the moat deepens |
| "Opinionated workflows beat generic flexibility" | Specific framework (trajectory + causal + cross-topic → projection) not generic "predict anything" |
| "Every output should explain itself" | Mandatory data citations, confidence levels, and signposts |
| Phase 3: "helps me decide, not just stay informed" | Direct implementation of this vision |
| ICP: "founders tracking AI, competitors, policy" | Thread trajectory + cross-topic signals directly serve this persona |

---

## 9. One-Paragraph Summary

Nexus should add projection capabilities in a phased approach, starting with thread trajectory analysis and cross-topic entity bridging (zero LLM cost, pure analytics on existing data), then adding causal chain extraction and a forward-looking "Projection" artifact in briefings. MiroFish's agent simulation approach is inspiring but unproven and expensive — use as directional inspiration only. Graphiti's bi-temporal model is the strongest external concept to adopt, but should be ported into SQLite queries rather than integrated as a dependency. The market gap between institutional intelligence tools ($10K+/mo) and consumer sentiment tools (<$250/mo) is real and growing, and Nexus's persistent knowledge base is the right foundation to build on. Start with Phase A (pure analytics), validate with users, and proceed through Phases B-D based on demonstrated value and accumulated data depth.
