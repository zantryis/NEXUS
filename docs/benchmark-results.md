# Nexus Experiment Report
**Date**: 2026-03-14T00:36:30.415871
**Duration**: 3026s
**Cost**: Gemini $0.18, DeepSeek $0.00

## Suite H: Divergence Prompt Variants
**Claim**: Broadened divergence instructions improve divergence_detection by X%

| Variant | completeness | source_balance | convergence_accuracy | divergence_detection | entity_coverage | overall |
|---------|------|------|------|------|------|------|
| div_baseline | 8.3 +/- 0.5 | 7.3 +/- 2.5 | 5.3 +/- 0.9 | 2.0 +/- 0.0 | 7.3 +/- 0.9 | 5.6 +/- 1.5 |
| div_broadened | 6.0 +/- 2.8 | 4.7 +/- 2.5 | 4.0 +/- 1.6 | 2.0 +/- 0.0 | 6.0 +/- 2.8 | 4.5 +/- 1.9 |
| div_structured | 8.0 +/- 1.6 | 6.0 +/- 3.3 | 5.3 +/- 2.5 | 4.0 +/- 1.6 | 8.0 +/- 0.0 | 6.3 +/- 1.8 |
| div_encouraged | 7.3 +/- 0.9 | 6.0 +/- 2.8 | 6.7 +/- 0.9 | 2.0 +/- 0.0 | 7.3 +/- 0.9 | 5.9 +/- 1.0 |

## Limitations
- LLM-as-judge evaluation (not human annotation)
- Single time snapshot (2026-03-14)
- N=4 topics
- Article availability depends on RSS feed state at time of polling
