# Nexus Experiment Report
**Date**: 2026-03-14T01:48:04.712017
**Duration**: 3099s
**Cost**: Gemini $0.19, DeepSeek $0.00

## Suite H: Divergence Prompt Variants
**Claim**: Broadened divergence instructions improve divergence_detection by X%

| Variant | completeness | source_balance | convergence_accuracy | divergence_detection | entity_coverage | overall |
|---------|------|------|------|------|------|------|
| div_baseline | 8.0 +/- 1.6 | 6.0 +/- 2.8 | 3.3 +/- 0.9 | 2.0 +/- 0.0 | 8.7 +/- 0.9 | 5.6 +/- 1.2 |
| div_broadened | 7.3 +/- 0.9 | 6.7 +/- 2.5 | 4.0 +/- 1.6 | 2.0 +/- 0.0 | 8.0 +/- 0.0 | 5.6 +/- 1.0 |
| div_structured | 8.7 +/- 0.9 | 7.3 +/- 0.9 | 6.7 +/- 1.9 | 3.3 +/- 0.9 | 8.7 +/- 0.9 | 6.9 +/- 0.8 |
| div_encouraged | 8.7 +/- 0.9 | 6.3 +/- 2.6 | 7.0 +/- 0.8 | 2.0 +/- 0.0 | 8.0 +/- 0.0 | 6.4 +/- 0.9 |

## Limitations
- LLM-as-judge evaluation (not human annotation)
- Single time snapshot (2026-03-14)
- N=4 topics
- Article availability depends on RSS feed state at time of polling
