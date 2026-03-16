# Kalshi Benchmark Results

**Dataset**: 500 settled markets from Kalshi
**Date**: 2026-03-16
**Primary engine**: structural (reasoning-first, 3 LLM calls)

## Known Limitations

1. **Market baseline confound**: `market_prob_at_cutoff` is the settlement price (near 0/1),
   NOT a pre-resolution snapshot. All "vs market" comparisons in this document are therefore
   meaningless. A proper forward-looking benchmark with daily price snapshots is needed.
2. **Hindsight bias**: LLM engines may "remember" outcomes from training data. Absolute
   accuracy numbers may be inflated. Engine-vs-engine comparisons are still fair (same LLM).
3. **Empty knowledge graph**: The KG had 0 events/entities/relationships at benchmark time.
   All predictions used pure LLM world knowledge. Evidence assembly ran but found nothing.
4. **Extreme skew**: 81% of outcomes were NO. Naive "always NO" gets 78.4% accuracy.

## Structural Engine (Primary — Reasoning-First)

**Architecture**: 3 LLM calls (base rate analyst -> contrarian -> supervisor reconciliation).
Outputs verdict (yes/no/uncertain) + confidence (high/medium/low), NOT raw probability floats.
Runs **independent** — no market anchor, no market probability passed to the engine.

### Fact-Based Evaluation (500 questions, full dataset)

| Metric | Value |
|--------|-------|
| **Coverage** | 80.6% (403/500 called, 60 uncertain, 37 JSON parse fallback) |
| **Accuracy** | 80.9% (326/403) |
| "Always NO" naive baseline | 78.4% |
| Value-add vs naive | +2.5pp |
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
4. **37 JSON parse failures** (7.4%): Gemini occasionally returns arrays instead of objects.
   Fixed in code, would improve on rerun.

## Engine Comparison (Brier — relative only)

These Brier scores used the confounded market baseline. The **relative ordering between
engines** is still valid (same LLM, same confound), but absolute values and "vs market"
numbers should be disregarded.

**Qualitative findings** (n=38 mid-range + near-extreme questions):
- Knowledge-augmented engines (actor, graphrag) performed best
- Naked LLM (zero context) performed worst
- Multi-persona debate added no value over independent reasoning (+0.0005 Brier, 5 extra LLM calls)
- Structured knowledge retrieval outperforms unstructured multi-persona approaches

## Dataset Characteristics

- **Outcomes**: YES=94, NO=406
- **Probability distribution**: Extreme=462, Mid-range=12, Other=26
- **Knowledge coverage**:
  - uncovered: 264
  - ai-ml-research: 206
  - iran-us-relations: 17
  - global-energy-transition: 13

## Next Steps

1. **Forward-looking benchmark**: Daily Kalshi price snapshots + structural engine predictions
   on open markets. Score when they settle. Eliminates both hindsight bias and market confound.
2. **Populate knowledge graph**: Run pipeline to get events/entities/relationships flowing,
   then test whether evidence assembly actually improves predictions.
3. **More knowledge coverage**: Most Kalshi categories don't overlap with existing Nexus topics.
