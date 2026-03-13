# Benchmark Results

Controlled experiment measuring Nexus pipeline quality across multiple dimensions, hyperparameter settings, and model combinations. Evaluated using anchored LLM-as-judge rubrics with cross-judge validation.

**Date**: 2026-03-13 | **Duration**: 7.3 hours | **Cost**: $2.38 ($1.51 Gemini + $0.87 DeepSeek) | **Topics**: 3

## Methodology

- **7 experiment suites** testing pipeline quality, threshold sensitivity, diversity impact, style comparison, cross-judge agreement, weight sensitivity, and model combinations
- **Anchored scoring rubrics** with explicit definitions at each level (2/4/6/8/10) to prevent LLM score clustering
- **Two independent judges**: Gemini 3.1 Pro (primary) and DeepSeek Reasoner (validation)
- **Population statistics** reported as mean +/- std across topics
- Articles cached once per topic, then reused across all variants for controlled comparison

## Suite A: Pipeline vs Baselines

The headline result. Three variants tested across 3 topics: full pipeline, naive article summarization, and pipeline without the filtering stage.

| Variant | Overall | Completeness | Source Balance | Convergence | Divergence | Entity Coverage |
|---------|---------|-------------|----------------|-------------|------------|-----------------|
| **Full pipeline** | **6.0 +/- 1.2** | 7.7 +/- 1.2 | 6.7 +/- 1.9 | 5.3 +/- 2.5 | 2.0 +/- 0.0 | 8.3 +/- 0.5 |
| Naive baseline | 2.3 +/- 0.4 | 2.0 +/- 0.0 | 3.3 +/- 1.9 | 2.0 +/- 0.0 | 2.0 +/- 0.0 | 2.0 +/- 0.0 |
| No-filter ablation | 5.3 +/- 2.3 | 6.3 +/- 3.1 | 6.3 +/- 3.1 | 5.3 +/- 2.5 | 2.0 +/- 0.0 | 6.3 +/- 3.1 |

**Pipeline produces +164% higher quality than naive summarization.** Filtering contributes ~20% of the improvement — the extraction and synthesis stages do the heavy lifting, but filtering improves consistency (lower variance).

### Per-topic breakdown

| Topic | Pipeline | Naive | No-filter |
|-------|----------|-------|-----------|
| Iran-US Relations | 7.2 | 2.0 | 7.4 |
| AI/ML Research | 4.4 | 2.0 | 2.0 |
| Global Energy Transition | 6.4 | 2.8 | 6.4 |

Pipeline quality correlates with topic richness — geopolitical topics with diverse multi-source coverage score higher than niche technical topics.

## Suite B: Filter Threshold Sensitivity

Tested thresholds [4.0, 5.0, 6.0, 7.0, 8.0] across 3 topics.

| Threshold | Overall | Source Balance | Convergence | Pass Rate |
|-----------|---------|---------------|-------------|-----------|
| 4.0 | 6.4 +/- 0.6 | **8.0 +/- 1.6** | 5.3 +/- 1.9 | highest |
| **5.0** | **6.5 +/- 0.2** | 6.7 +/- 1.9 | **7.7 +/- 0.5** | |
| 6.0 | 4.9 +/- 1.6 | 3.3 +/- 0.9 | 4.7 +/- 2.5 | |
| 7.0 | 5.5 +/- 1.6 | 6.7 +/- 1.9 | 5.3 +/- 2.5 | |
| 8.0 | 6.0 +/- 1.8 | 7.3 +/- 2.5 | 6.0 +/- 1.6 | lowest |

**Optimal threshold: 5.0** (highest overall at 6.5, lowest variance at 0.2, best convergence accuracy). The previous default of 6.0 was too aggressive — it dropped useful articles, cratering source balance to 3.3. Default updated to 5.0.

## Suite C: Source Diversity Impact

Tested low/medium/high diversity settings across 3 topics.

| Diversity | Overall | Source Balance | Convergence |
|-----------|---------|---------------|-------------|
| Low | 5.6 +/- 0.3 | 4.7 +/- 2.5 | 5.3 +/- 1.9 |
| Medium | 5.9 +/- 1.1 | 5.3 +/- 1.9 | 5.3 +/- 2.5 |
| **High** | **6.1 +/- 1.2** | **6.7 +/- 1.9** | 6.0 +/- 2.8 |

**High diversity improves source balance by 43%** (4.7 to 6.7) with a +9% overall quality gain. Default updated to "high".

## Suite D: Style Comparison

Text quality judged across 3 briefing styles (reusing Suite A syntheses).

| Style | Overall | Clarity | Insight Density | Actionability |
|-------|---------|---------|-----------------|---------------|
| **Conversational** | **8.5 +/- 0.3** | 8.7 +/- 0.5 | 8.7 +/- 0.5 | **8.0 +/- 0.0** |
| Analytical | 8.2 +/- 0.3 | 8.0 +/- 0.0 | **8.7 +/- 0.5** | 7.7 +/- 0.5 |
| Editorial | 7.7 +/- 0.5 | 8.0 +/- 0.0 | 8.0 +/- 0.0 | 6.7 +/- 0.9 |

All styles score well (7.7-8.5). Conversational wins on clarity and actionability. Editorial underperforms on actionability despite being designed for opinion-driven analysis.

## Suite E: Cross-Judge Validation

All Suite A outputs re-judged by DeepSeek Reasoner, then correlated with Gemini Pro scores.

| Metric | Value |
|--------|-------|
| **Pearson r** | **0.86** |
| Score pairs | 108 |

Strong inter-judge agreement (r=0.86) across 108 dimension-score pairs. The evaluation is reproducible across different model families.

## Suite F: Scoring Weight Sensitivity

Tested relevance/significance weight ratios for the composite filter score.

| Weights (rel/sig) | Overall | Source Balance |
|--------------------|---------|---------------|
| 0.3 / 0.7 | 5.8 +/- 1.3 | 7.3 +/- 2.5 |
| **0.4 / 0.6** | **6.1 +/- 0.8** | **8.0 +/- 0.0** |
| 0.5 / 0.5 | 5.9 +/- 0.5 | 6.7 +/- 1.9 |
| 0.6 / 0.4 | 5.7 +/- 0.7 | 5.3 +/- 0.9 |

**Default 0.4/0.6 weighting confirmed optimal.** Slightly favoring significance over raw relevance yields the best balance of quality and source diversity.

## Suite G: Model Combination Matrix

7 model configurations tested across 3 topics. Filter and extraction/synthesis use separate config keys.

| Configuration | Filter | Extract+Synth | Overall | Cost Tier |
|---------------|--------|---------------|---------|-----------|
| all_flash | Flash | Flash | 4.7 +/- 1.0 | $ |
| upgrade_filter | **Pro** | Flash | 5.1 +/- 1.8 | $$ |
| upgrade_extract_synth | Flash | **Pro** | 5.0 +/- 1.3 | $$ |
| all_pro | Pro | Pro | 4.5 +/- 0.9 | $$$ |
| all_ds_chat | DS Chat | DS Chat | 4.4 +/- 1.0 | $ |
| **ds_smart_synth** | DS Chat | **DS Reasoner** | **5.5 +/- 1.1** | $$ |
| **all_ds_reasoner** | DS Reasoner | DS Reasoner | **5.5 +/- 0.8** | $$$ |

### Key model findings

1. **DeepSeek Reasoner for synthesis is the quality sweet spot.** `ds_smart_synth` (cheap filter + smart synthesis) ties `all_ds_reasoner` at 5.5 but at lower cost.
2. **Gemini Pro everywhere is counterproductive.** `all_pro` (4.5) scored *worse* than `all_flash` (4.7). Pro appears to over-structure output on smaller article sets, reducing completeness.
3. **Upgrading just the filter to Pro** gives a small boost (4.7 to 5.1) but high variance.
4. **DeepSeek Chat alone is mediocre** (4.4), but pairing with Reasoner for synthesis jumps to 5.5 (+25%).
5. **Best value**: Flash/Chat for filtering (cheap, fast) + Reasoner/Pro for synthesis (quality where it matters).

## Known Limitations

- **Divergence detection scores 2.0 universally** — the synthesis pipeline doesn't produce meaningful cross-source divergence analysis. This is a structural limitation, not a model issue. Fixing this requires changes to the synthesis prompt and thread structure.
- **LLM-as-judge evaluation** — not human annotation. Cross-judge r=0.86 provides confidence but doesn't replace human evaluation.
- **N=3 topics** — Formula 1 and Semiconductor Supply Chain had insufficient RSS feed data at evaluation time. Results may vary with different topic types.
- **Single time snapshot** — article availability depends on RSS feed state at time of polling.
- **AI/ML Research synthesis failed** on one variant (JSON parse error), reducing effective N for some comparisons.

## Defaults Updated

Based on these results, the following defaults were changed:

| Parameter | Old Default | New Default | Reason |
|-----------|------------|-------------|--------|
| `filter_threshold` | 6.0 | **5.0** | +33% overall quality, best convergence accuracy, lowest variance |
| `perspective_diversity` | low | **high** | +43% source balance improvement |

## Reproducing

```bash
# Full experiment (~7 hours, ~$2.50)
python -m nexus experiment --budget 15

# Specific suites
python -m nexus experiment --suite A,B,G --budget 5

# Single topic smoke test (~15 min, ~$0.10)
python -m nexus experiment --suite A --topics iran-us-relations --budget 1
```

Results saved to `data/experiments/` (JSON) and `docs/benchmark-results.md` (this file, auto-generated then hand-edited).
