# Leakage-Safe Replay Benchmark: 2026-03-11 to 2026-03-15

This note captures the current strict, leakage-safe forecast benchmark after the replay firewall landed and the forecast catalog was expanded beyond pure event-count continuation.

- Leakage audit command:
  - `python -m nexus forecast audit-leakage --start 2026-03-11 --end 2026-03-15 --profile signal-rich`
- Strict replay benchmark command:
  - `python -m nexus forecast benchmark --engines native,baseline,trajectory --start 2026-03-11 --end 2026-03-15 --mode replay --profile signal-rich`
- Scope:
  - signal-rich only
  - deterministic only
  - live benchmark reads from an isolated SQLite snapshot, not the live `knowledge.db`

## What the audit found

The legacy replay path was not safe enough to trust for forecasting claims.

- `10` audited cutoffs
- `8` cutoffs showed present-day thread snapshot contamination
- `8` cutoffs showed future cross-topic signal contamination
- audited topics: `ai-ml-research`, `global-energy-transition`

Representative failures:

- AI/ML cutoffs on March 11-14, 2026 were pulling latest thread snapshot counts from March 15, 2026 instead of the cutoff state
- AI/ML and energy cutoffs were seeing cross-topic bridge rows observed on March 14-15, 2026 while evaluating March 11-13, 2026 cutoffs

That means the old replay path was useful as an infrastructure check, but not valid for quality claims.

## Strict replay results

| Engine | Total | Accuracy | Mean Brier | Mean Log Loss |
| --- | ---: | ---: | ---: | ---: |
| `native` | 12 | 0.333 | 0.5701 | 1.8320 |
| `baseline` | 12 | 0.333 | 0.2854 | 0.7664 |
| `trajectory` | 12 | 0.333 | 0.4515 | 1.1652 |

Calibration snapshot:

- `native`: `6` forecasts in `0.40-0.60` with `0.667` hit rate, but `6` in `0.80-0.95` with `0.0` hit rate
- `baseline`: `6` forecasts in `0.20-0.40` with `0.667` hit rate, `6` in `0.40-0.60` with `0.0` hit rate
- `trajectory`: `6` forecasts in `0.40-0.60` with `0.667` hit rate, higher buckets still overconfident

Benchmark metadata:

- cutoff count: `10`
- domains: `ai-ml-research`, `global-energy-transition`
- resolved forecasts: `12`
- verdict: `infrastructure-valid, statistically-insufficient`

## Interpretation (pre-calibration fix)

This is a real result, and it is better than the first strict replay run, but it is still not a model-quality win.

- The good news: the strict benchmark path works, does not mutate the live DB, and the catalog expansion increased resolved forecasts from `7` to `12`.
- The important bad news: the deterministic native engine still loses to the base-rate baseline on Brier score and log loss because its high-confidence bucket is badly calibrated.
- The broader warning: the current corpus is still too small and overfiltered for statistically credible claims.

---

## Calibration Fix Results: 2026-03-10 to 2026-03-15

After probability calibration (lowered trajectory base rates, tightened momentum boosts, evidence-proportional cross-topic scoring) and lowering `min_thread_snapshots` from 3→2 to include iran-us-relations:

| Engine | Total | Accuracy | Mean Brier | Mean Log Loss |
| --- | ---: | ---: | ---: | ---: |
| `native` | 25 | 0.440 | 0.3564 | 0.9170 |
| `baseline` | 25 | 0.440 | 0.3160 | 0.8302 |
| `trajectory` | 25 | 0.440 | **0.3118** | **0.8177** |

Calibration snapshot:

- `native`: `13` in `0.40-0.60` (0.846 hit rate), `12` in `0.60-0.80` (0.0 hit rate)
- `baseline`: `13` in `0.20-0.40` (0.846 hit rate), `12` in `0.40-0.60` (0.0 hit rate)
- `trajectory`: all `25` in `0.40-0.60` (0.44 hit rate)

Benchmark metadata:

- cutoff count: `16`
- domains: `ai-ml-research`, `global-energy-transition`, `iran-us-relations`
- resolved forecasts: `25`
- verdict: `infrastructure-valid, statistically-insufficient`

### What changed

| Parameter | Before | After |
|---|---|---|
| `about_to_break` base rate | 0.78 | 0.58 |
| `accelerating` base rate | 0.66 | 0.50 |
| `steady` base rate | 0.48 | 0.38 |
| `decelerating` base rate | 0.32 | 0.25 |
| Native thread momentum | [-0.12, +0.18]/20 | [-0.08, +0.10]/30 |
| Cross-topic boost | flat +0.12 | min(0.06, len(event_ids)*0.02) |
| `min_thread_snapshots` default | 3 | 2 |

### Interpretation (post-calibration)

- The `trajectory` engine now **beats baseline** on both Brier (0.3118 vs 0.316) and log loss (0.8177 vs 0.8302).
- The `native` engine improved dramatically (0.5701 → 0.3564) but still trails baseline due to 12 forecasts in `0.60-0.80` with 0% hit rate. These are momentum-boosted `about_to_break` threads.
- The catastrophic `0.80-0.95` bucket is completely eliminated.
- Iran-US-relations is now included, adding `6` cutoffs and richer signal data.

### What this means next

- The remaining calibration target is the `0.60-0.80` bucket. Options: lower `about_to_break` base rate further to 0.50, or cap the total momentum ceiling at 0.58.
- The corpus is still too small for statistical significance. Next milestone: `100+` resolved forecasts via larger date range or more topics.
- The trajectory engine is the current best performer and could become the default if native doesn't close the gap.
