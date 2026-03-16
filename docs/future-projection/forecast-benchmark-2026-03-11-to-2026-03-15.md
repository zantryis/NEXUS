# Forecast Benchmark: 2026-03-11 to 2026-03-15

This note captures the first quantified replay benchmark after the forecast engine landed on `feature/future-projection-alpha`.

- Command: `python -m nexus forecast benchmark --engines native,baseline,kuzu --start 2026-03-11 --end 2026-03-15`
- Source artifact: `data/benchmarks/forecast-benchmark-2026-03-11-2026-03-15.json`
- Forecast types exercised: `binary`
- Resolution families exercised: `thread_new_event_count`

## Results

| Engine | Total | Accuracy | Mean Brier | Mean Log Loss |
| --- | ---: | ---: | ---: | ---: |
| `native` | 7 | 0.0 | 0.9025 | 2.9957 |
| `baseline` | 7 | 0.0 | 0.6561 | 1.6607 |
| `kuzu` | 7 | 0.0 | 0.9025 | 2.9957 |

Calibration snapshot:

- `native`: `0.80-0.95` bucket, `7` forecasts, `0` resolved true
- `baseline`: `0.80-0.95` bucket, `7` forecasts, `0` resolved true
- `kuzu`: `0.80-0.95` bucket, `7` forecasts, `0` resolved true

## Interpretation

These numbers are operationally useful but not yet scientifically strong.

- The replay, scoring, and no-fake-resolution pipeline works end to end.
- The current benchmark window is too sparse and too overfiltered to support strong quality claims.
- The native engine currently overweights short-horizon event-count continuations in volatile topics.
- The `kuzu` sidecar path is still a graph-enhanced heuristic variant, not a full Kuzu-backed retrieval engine, so parity with `native` is expected here.

## Immediate implications

- Forecast generation is now measurable with Brier score, log loss, and calibration buckets.
- The next work should improve forecast target diversity and backfill richer, cleaner historical windows.
- This artifact should be treated as a baseline infrastructure check, not as evidence that forecast quality is already competitive.
