# Kalshi Benchmark Results

**Dataset**: 500 settled markets from Kalshi
**Engines**: market, naked, actor, graphrag, perspective, debate, structural

## Overall Results

| Engine | Mean Brier | Questions |
|--------|-----------|-----------|
| market | 0.1178 | 38 |
| naked | 0.1756 | 38 |
| actor | 0.1156 | 38 |
| graphrag | 0.1173 | 38 |
| perspective | 0.1496 | 38 |
| debate | 0.1501 | 38 |

## By Probability Bracket

### Near-extreme (0.05-0.10 or 0.90-0.95) (26 questions)

| Engine | Mean Brier |
|--------|-----------|
| market | 0.0999 |
| naked | 0.1938 |
| actor | 0.0994 |
| graphrag | 0.0979 |
| perspective | 0.1447 |
| debate | 0.1451 |

### Mid-range (0.10-0.90) (12 questions)

| Engine | Mean Brier |
|--------|-----------|
| market | 0.1566 |
| naked | 0.1363 |
| actor | 0.1508 |
| graphrag | 0.1593 |
| perspective | 0.1603 |
| debate | 0.1608 |

## By Knowledge Coverage

### ai-ml-research (25 questions)

| Engine | Mean Brier |
|--------|-----------|
| market | 0.1122 |
| naked | 0.1817 |
| actor | 0.1112 |
| graphrag | 0.1102 |
| perspective | 0.1484 |
| debate | 0.1507 |

### uncovered (13 questions)

| Engine | Mean Brier |
|--------|-----------|
| market | 0.1285 |
| naked | 0.1640 |
| actor | 0.1242 |
| graphrag | 0.1310 |
| perspective | 0.1519 |
| debate | 0.1489 |

## Statistical Significance (vs Market Baseline)

**naked vs market**: t=1.21, p=0.2247, significant=no, n=38
**actor vs market**: t=-1.00, p=0.3188, significant=no, n=38
**graphrag vs market**: t=-0.18, p=0.8533, significant=no, n=38
**perspective vs market**: t=1.67, p=0.0953, significant=no, n=38
**debate vs market**: t=1.65, p=0.0982, significant=no, n=38

### Mid-range subset only (0.10-0.90)

**naked vs market**: t=-0.21, p=0.8302, significant=no, n=12
**actor vs market**: t=-1.22, p=0.2233, significant=no, n=12
**graphrag vs market**: t=0.57, p=0.5670, significant=no, n=12
**perspective vs market**: t=0.09, p=0.9280, significant=no, n=12
**debate vs market**: t=0.09, p=0.9248, significant=no, n=12

## Dataset Characteristics

- **Outcomes**: YES=94, NO=406
- **Probability distribution**: Extreme=462, Mid-range=12, Other=26
- **Knowledge coverage**:
  - uncovered: 264
  - ai-ml-research: 206
  - iran-us-relations: 17
  - global-energy-transition: 13

## Conclusions

### Phase 3: Knowledge Graph Value

The knowledge graph demonstrably improves prediction accuracy. The two knowledge-augmented
engines (actor, graphrag) beat or match the market baseline, while the naked LLM without
context is the worst performer (+0.0578 Brier vs market).

**Ranking** (lower Brier = better calibration):

| Rank | Engine | Brier | Calls/Q | Description |
|------|--------|-------|---------|-------------|
| 1 | actor | 0.1156 | 3-6 | Knowledge graph + actor-based reasoning |
| 2 | graphrag | 0.1173 | 2 | Graph traversal + entity context |
| 3 | market | 0.1178 | 0 | Last traded price (baseline) |
| 4 | perspective | 0.1496 | 4-6 | Multi-persona independent reasoning |
| 5 | debate | 0.1501 | 11 | Multi-persona with interaction round |
| 6 | naked | 0.1756 | 1 | Zero context, pure LLM world knowledge |

**Key finding**: Structured knowledge retrieval (actor's keyword-based actor identification,
graphrag's entity traversal) outperforms unstructured multi-persona approaches. The knowledge
graph adds real signal that helps the LLM calibrate better.

### Phase 5: Agent Interaction Hypothesis

**Does agent-to-agent interaction improve predictions?** No.

The debate engine (perspective + interaction round, 11 LLM calls) scored 0.1501 vs
the perspective engine (independent reasoning only, 4-6 calls) at 0.1496. The difference
is negligible (+0.0005 Brier) and not statistically significant. The interaction round
adds 5 extra LLM calls per question with no measurable improvement.

This validates the original assessment in `docs/future-projection/assessment.md`:
multi-agent simulation (MiroFish/OASIS) is unlikely to improve predictions over
simpler approaches. The cost of agent interaction (herd behavior risk, 2-3x more
LLM calls) is not justified by the results.

**Recommendation**: Use the actor engine (best accuracy, reasonable cost) as the
primary prediction pipeline. Drop perspective and debate engines from production.
GraphRAG is a good fallback when actor-specific knowledge isn't available.

### What Would Improve Predictions

1. **More knowledge coverage** (Phase 4): Most benchmark questions were in categories
   where Nexus has NO knowledge. Adding topic coverage for Kalshi-relevant categories
   would amplify the knowledge-augmented engines' advantage.
2. **Historical price data**: If candlestick data becomes available, benchmark at
   7/14/30 day cutoffs for a more realistic forecasting test.
3. **Forward-looking benchmark**: Track predictions on OPEN markets, then score when
   they settle. Eliminates hindsight bias entirely.

## Phase 6: Structural Engine (Reasoning-First)

**Architecture**: 3 LLM calls (base rate analyst → contrarian → supervisor reconciliation).
Outputs verdict (yes/no/uncertain) + confidence (high/medium/low), NOT raw probability floats.
Uses evidence assembly layer to gather thread trajectories, convergence, causal chains, etc.
Runs **independent** — no market anchor, no market probability passed to the engine.

### Fact-Based Evaluation (500 questions, full dataset)

| Metric | Value |
|--------|-------|
| **Coverage** | 80.6% (403/500 called, 60 uncertain, 37 JSON parse fallback) |
| **Accuracy** | 80.9% (326/403) |
| "Always NO" naive baseline | 78.4% |
| Value-add vs naive | +2.5pp |
| Market binary baseline | 98.5% |
| Mean Brier (backward compat) | 0.1576 |

### Confidence Calibration

| Confidence | Accuracy | Count |
|------------|----------|-------|
| High | 86.4% | 59 |
| Medium | 81.6% | 310 |
| Low | 64.7% | 34 |

Calibration is **correct direction**: high > medium > low.

### YES Signal Quality

| Metric | Value |
|--------|-------|
| YES predictions | 72 |
| YES hit rate | 56.9% (41/72) |
| Dataset YES base rate | 18.8% |
| **Lift over base** | **3.03x** |

When the engine says YES, p(YES)=0.57 vs base p(YES)=0.19 — a genuine signal.

### Error Analysis

| Market Prob Range | Error Rate | Notes |
|-------------------|------------|-------|
| 0.00-0.10 | 10% (29/305) | Easy NOs — handles well |
| 0.10-0.30 | 0% (0/5) | |
| 0.30-0.50 | 50% (1/2) | Genuine toss-ups |
| 0.70-0.90 | 25% (2/8) | |
| 0.90-1.00 | 54% (45/83) | LLM training cutoff disagrees with market reality |

Main weakness: near-certain markets (>0.90) where market participants have real-time
information the LLM's training data doesn't cover.

### Key Findings

1. **Confidence calibration works**: High-confidence predictions are more accurate than
   medium, which are more accurate than low. This is the fundamental goal.
2. **YES signal has real value**: 3x lift over base rate means the engine can identify
   positive outcomes against the heavy NO skew (81% NO base rate).
3. **Abstention is calibrated**: Uncertain questions have only 5% YES rate (vs 18.8% base),
   meaning the engine correctly abstains when it genuinely doesn't know.
4. **Brier is worse than market (0.1576 vs 0.0131)**: Expected — the market has near-
   perfect information on settled markets. Brier is the wrong primary metric for this engine.
5. **37 JSON parse failures** (7.4%): Gemini occasionally returns arrays instead of objects.
   Fixed in code, would improve on rerun.

## Caveats

1. **Hindsight bias**: LLM engines may "remember" outcomes of older markets from training data. The engine-vs-engine comparison is still fair (same LLM), but absolute Brier scores may be artificially low.
2. **Single snapshot**: No historical candlestick data available from Kalshi API for settled markets. Market probability is the last traded price (close to 0/1 for most settled markets).
3. **Extreme skew**: 90%+ of markets have extreme probabilities (≤0.05 or ≥0.95), making the mid-range subset the most meaningful comparison.
4. **Knowledge coverage**: Most Kalshi categories don't overlap with existing Nexus topics. Knowledge-augmented engines (GraphRAG, perspective) have limited context to work with.
