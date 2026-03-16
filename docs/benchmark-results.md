# Nexus Experiment Report
**Date**: 2026-03-14T17:17:44.889037
**Duration**: 32191s
**Cost**: Gemini $1.78, DeepSeek $0.91
**Environment**: local
**Fixtures**: live_polling

## Suite A: Pipeline vs Baselines
**Claim**: Nexus pipeline produces X% higher quality than naive summarization

| Variant | completeness | source_balance | convergence_accuracy | divergence_detection | entity_coverage | overall |
|---------|------|------|------|------|------|------|
| full_pipeline | 6.3 +/- 3.1 | 5.3 +/- 2.5 | 5.3 +/- 2.5 | 2.3 +/- 0.5 | 6.7 +/- 1.9 | 5.2 +/- 2.0 |
| naive_baseline | 3.0 +/- 1.0 | 3.0 +/- 1.0 | 2.0 +/- 0.0 | 2.0 +/- 0.0 | 3.0 +/- 1.0 | 2.6 +/- 0.6 |
| no_filter | 8.5 +/- 0.5 | 8.0 +/- 0.0 | 6.0 +/- 2.0 | 3.0 +/- 1.0 | 8.0 +/- 0.0 | 6.7 +/- 0.7 |

## Suite B: Filter Threshold Sensitivity
**Claim**: Optimal filter threshold is X.0 (stable across topic types)

| Variant | completeness | source_balance | convergence_accuracy | divergence_detection | entity_coverage | overall |
|---------|------|------|------|------|------|------|
| threshold_4.0 | 8.7 +/- 0.9 | 6.0 +/- 1.6 | 6.0 +/- 1.6 | 2.7 +/- 0.9 | 8.7 +/- 0.9 | 6.4 +/- 0.3 |
| threshold_5.0 | 9.0 +/- 0.8 | 6.0 +/- 1.6 | 5.7 +/- 1.7 | 2.7 +/- 0.9 | 7.3 +/- 2.5 | 6.1 +/- 0.7 |
| threshold_6.0 | 8.0 +/- 1.6 | 7.3 +/- 0.9 | 5.3 +/- 2.5 | 2.0 +/- 0.0 | 8.0 +/- 0.0 | 6.1 +/- 0.8 |
| threshold_7.0 | 6.0 +/- 2.8 | 4.0 +/- 1.6 | 2.7 +/- 0.9 | 2.0 +/- 0.0 | 6.0 +/- 2.8 | 4.1 +/- 1.5 |
| threshold_8.0 | 4.7 +/- 2.5 | 4.7 +/- 2.5 | 4.0 +/- 1.6 | 2.0 +/- 0.0 | 5.3 +/- 2.5 | 4.1 +/- 1.5 |

## Suite C: Source Diversity Impact
**Claim**: High diversity improves perspective balance by X%

| Variant | completeness | source_balance | convergence_accuracy | divergence_detection | entity_coverage | overall |
|---------|------|------|------|------|------|------|
| diversity_low | 6.0 +/- 2.8 | 4.7 +/- 2.5 | 4.0 +/- 1.6 | 2.0 +/- 0.0 | 6.7 +/- 1.9 | 4.7 +/- 1.7 |
| diversity_medium | 7.0 +/- 2.2 | 6.0 +/- 2.8 | 5.3 +/- 2.5 | 4.0 +/- 1.6 | 7.3 +/- 0.9 | 5.9 +/- 2.0 |
| diversity_high | 5.0 +/- 2.9 | 4.7 +/- 2.5 | 3.3 +/- 1.9 | 3.0 +/- 1.4 | 5.0 +/- 2.9 | 4.2 +/- 2.3 |

## Suite D: Style Comparison
**Claim**: Editorial style scores highest on actionability

| Variant | clarity | insight_density | source_attribution | narrative_coherence | actionability | overall |
|---------|------|------|------|------|------|------|
| analytical | 8.0 +/- 0.0 | 9.0 +/- 0.0 | 9.3 +/- 0.5 | 9.0 +/- 0.0 | 7.7 +/- 0.5 | 8.6 +/- 0.2 |
| conversational | 8.3 +/- 0.5 | 8.7 +/- 0.5 | 9.0 +/- 0.8 | 8.7 +/- 0.5 | 7.7 +/- 0.5 | 8.5 +/- 0.3 |
| editorial | 8.0 +/- 0.0 | 8.0 +/- 0.0 | 9.3 +/- 0.9 | 9.0 +/- 0.8 | 7.7 +/- 0.5 | 8.4 +/- 0.3 |

## Suite E: Cross-Judge Validation
**Claim**: Inter-judge agreement: Pearson r = 0.71

| Variant | pearson_r | n_pairs |
|---------|------|------|
| correlation | 0.7083 | 96 |

## Suite F: Scoring Weight Sensitivity
**Claim**: Default 0.4/0.6 weighting is optimal

| Variant | completeness | source_balance | convergence_accuracy | divergence_detection | entity_coverage | overall |
|---------|------|------|------|------|------|------|
| w_0.3_0.7 | 6.7 +/- 1.9 | 6.0 +/- 1.6 | 4.7 +/- 1.9 | 3.3 +/- 0.9 | 7.3 +/- 0.9 | 5.6 +/- 1.4 |
| w_0.4_0.6 | 4.0 +/- 2.8 | 4.0 +/- 1.6 | 3.3 +/- 1.9 | 2.0 +/- 0.0 | 4.7 +/- 2.5 | 3.6 +/- 1.7 |
| w_0.5_0.5 | 6.7 +/- 1.9 | 4.0 +/- 1.6 | 4.0 +/- 1.6 | 2.0 +/- 0.0 | 6.7 +/- 1.9 | 4.7 +/- 1.3 |
| w_0.6_0.4 | 6.7 +/- 1.9 | 4.7 +/- 1.9 | 4.0 +/- 1.6 | 2.7 +/- 0.9 | 6.7 +/- 1.9 | 4.9 +/- 1.5 |

## Suite G: Model Combination Matrix
**Claim**: Stronger models at [stage] yield X% improvement

| Variant | completeness | source_balance | convergence_accuracy | divergence_detection | entity_coverage | overall |
|---------|------|------|------|------|------|------|
| all_flash_gemini_pro | 4.7 +/- 2.5 | 4.0 +/- 1.6 | 3.3 +/- 1.9 | 2.3 +/- 0.5 | 4.7 +/- 2.5 | 3.8 +/- 1.8 |
| all_flash_deepseek_chat | 6.7 +/- 0.9 | 3.3 +/- 0.9 | 3.3 +/- 1.9 | 3.3 +/- 1.9 | 6.7 +/- 0.9 | 4.7 +/- 0.7 |
| upgrade_filter_gemini_pro | 4.0 +/- 1.6 | 2.7 +/- 0.9 | 2.7 +/- 0.9 | 2.0 +/- 0.0 | 4.7 +/- 2.5 | 3.2 +/- 1.0 |
| upgrade_filter_deepseek_chat | 6.0 +/- 2.8 | 3.3 +/- 0.9 | 3.3 +/- 1.9 | 3.3 +/- 1.9 | 6.7 +/- 1.9 | 4.5 +/- 1.6 |
| upgrade_extract_synth_gemini_pro | 6.7 +/- 1.9 | 5.3 +/- 2.5 | 5.7 +/- 2.6 | 2.0 +/- 0.0 | 6.7 +/- 1.9 | 5.3 +/- 1.8 |
| upgrade_extract_synth_deepseek_chat | 6.7 +/- 1.9 | 4.0 +/- 1.6 | 4.7 +/- 1.9 | 4.7 +/- 1.9 | 6.7 +/- 1.9 | 5.3 +/- 1.8 |
| all_pro_gemini_pro | 7.0 +/- 1.0 | 4.0 +/- 0.0 | 6.0 +/- 2.0 | 2.0 +/- 0.0 | 8.0 +/- 0.0 | 5.4 +/- 0.6 |
| all_pro_deepseek_chat | 7.0 +/- 1.0 | 5.0 +/- 1.0 | 6.0 +/- 0.0 | 6.0 +/- 0.0 | 7.0 +/- 1.0 | 6.2 +/- 0.6 |
| all_ds_chat_gemini_pro | 8.7 +/- 0.9 | 8.0 +/- 1.6 | 6.0 +/- 1.6 | 2.7 +/- 0.9 | 8.7 +/- 0.9 | 6.8 +/- 1.2 |
| all_ds_chat_deepseek_chat | 7.7 +/- 0.5 | 4.7 +/- 0.9 | 6.0 +/- 0.0 | 7.3 +/- 0.9 | 8.0 +/- 0.0 | 6.7 +/- 0.4 |
| ds_smart_synth_gemini_pro | 7.3 +/- 0.9 | 6.7 +/- 1.9 | 6.0 +/- 1.6 | 2.0 +/- 0.0 | 7.3 +/- 0.9 | 4.1 +/- 0.5 |
| ds_smart_synth_deepseek_chat | 6.7 +/- 0.9 | 4.0 +/- 0.0 | 5.3 +/- 0.9 | 3.3 +/- 0.9 | 7.3 +/- 0.9 | 5.3 +/- 0.2 |
| all_ds_reasoner_gemini_pro | 6.7 +/- 2.5 | 8.0 +/- 1.6 | 6.0 +/- 1.6 | 2.7 +/- 0.9 | 7.3 +/- 0.9 | 6.1 +/- 1.4 |
| all_ds_reasoner_deepseek_chat | 6.7 +/- 0.9 | 4.7 +/- 0.9 | 5.3 +/- 0.9 | 2.7 +/- 0.9 | 6.7 +/- 0.9 | 5.2 +/- 0.7 |

## Suite H: Divergence Prompt Variants
**Claim**: Broadened divergence instructions improve divergence_detection by X%

| Variant | completeness | source_balance | convergence_accuracy | divergence_detection | entity_coverage | overall |
|---------|------|------|------|------|------|------|
| div_baseline | 4.7 +/- 2.5 | 4.0 +/- 1.6 | 3.3 +/- 0.9 | 2.0 +/- 0.0 | 6.0 +/- 1.6 | 4.0 +/- 1.3 |
| div_broadened | 6.0 +/- 2.8 | 5.3 +/- 2.5 | 3.3 +/- 0.9 | 2.0 +/- 0.0 | 6.7 +/- 1.9 | 4.7 +/- 1.6 |
| div_structured | 5.3 +/- 2.5 | 6.0 +/- 2.8 | 4.0 +/- 1.6 | 2.7 +/- 0.9 | 6.0 +/- 2.8 | 4.8 +/- 2.0 |
| div_encouraged | 7.0 +/- 2.4 | 6.7 +/- 2.5 | 4.0 +/- 1.6 | 2.0 +/- 0.0 | 8.7 +/- 0.9 | 5.7 +/- 1.5 |

## Limitations
- LLM-as-judge evaluation (not human annotation)
- Single time snapshot (2026-03-14)
- N=4 topics
- Article availability depends on RSS feed state at time of polling
