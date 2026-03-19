# Kalshi Benchmark Results

**Dataset**: 500 settled markets from Kalshi
**Engines**: market, naked, structural, actor, graphrag, perspective, debate

This document is part of the forecast lab. It is useful for calibration and internal comparison work, but it is not part of the default `v0.1` onboarding path.

## Overall Results

| Engine | Mean Brier | Questions |
|--------|-----------|-----------|
| market | 0.1575 | 114 |
| naked | 0.2059 | 114 |
| structural | 0.2354 | 114 |
| actor | 0.2301 | 114 |
| graphrag | 0.2374 | 114 |
| perspective | 0.2555 | 114 |
| debate | 0.2617 | 114 |

## By Probability Bracket

### Mid-range (0.10-0.90) (114 questions)

| Engine | Mean Brier |
|--------|-----------|
| market | 0.1575 |
| naked | 0.2059 |
| structural | 0.2354 |
| actor | 0.2301 |
| graphrag | 0.2374 |
| perspective | 0.2555 |
| debate | 0.2617 |

## By Knowledge Coverage

### ai-ml-research (25 questions)

| Engine | Mean Brier |
|--------|-----------|
| market | 0.1543 |
| naked | 0.1895 |
| structural | 0.1562 |
| actor | 0.2605 |
| graphrag | 0.2315 |
| perspective | 0.2014 |
| debate | 0.1808 |

### global-energy-transition (3 questions)

| Engine | Mean Brier |
|--------|-----------|
| market | 0.3359 |
| naked | 0.2466 |
| structural | 0.2771 |
| actor | 0.2502 |
| graphrag | 0.2403 |
| perspective | 0.2583 |
| debate | 0.2417 |

### iran-us-relations (9 questions)

| Engine | Mean Brier |
|--------|-----------|
| market | 0.2105 |
| naked | 0.0530 |
| structural | 0.0729 |
| actor | 0.0447 |
| graphrag | 0.0092 |
| perspective | 0.1456 |
| debate | 0.1467 |

### uncovered (77 questions)

| Engine | Mean Brier |
|--------|-----------|
| market | 0.1454 |
| naked | 0.2275 |
| structural | 0.2785 |
| actor | 0.2411 |
| graphrag | 0.2658 |
| perspective | 0.2857 |
| debate | 0.3022 |

## Statistical Significance (vs Market Baseline)

**naked vs market**: t=1.95, p=0.0514, significant=no, n=114
**structural vs market**: t=3.39, p=0.0007, significant=YES, n=114
**actor vs market**: t=3.19, p=0.0014, significant=YES, n=114
**graphrag vs market**: t=3.25, p=0.0012, significant=YES, n=114
**perspective vs market**: t=3.95, p=0.0001, significant=YES, n=114
**debate vs market**: t=3.91, p=0.0001, significant=YES, n=114

### Mid-range subset only (0.10-0.90)

**naked vs market**: t=1.95, p=0.0514, significant=no, n=114
**structural vs market**: t=3.39, p=0.0007, significant=YES, n=114
**actor vs market**: t=3.19, p=0.0014, significant=YES, n=114
**graphrag vs market**: t=3.25, p=0.0012, significant=YES, n=114
**perspective vs market**: t=3.95, p=0.0001, significant=YES, n=114
**debate vs market**: t=3.91, p=0.0001, significant=YES, n=114

## Dataset Characteristics

- **Outcomes**: YES=94, NO=406
- **Probability distribution**: Extreme=462, Mid-range=12, Other=26
- **Knowledge coverage**:
  - uncovered: 264
  - ai-ml-research: 206
  - iran-us-relations: 17
  - global-energy-transition: 13

## Gamma Sweep (post-hoc recalibration)

For each engine, undo the production gamma (0.8) and reapply with candidate gammas.

### naked (production gamma=0.8)

| Gamma | Mean Brier | vs Market |
|-------|-----------|-----------|
| 0.50 | 0.2015 | +0.0440 |
| 0.60 | 0.2014 | +0.0439 |
| 0.70 | 0.2031 | +0.0456 |
| 0.80 | 0.2059 | +0.0484 |
| 0.90 | 0.2094 | +0.0519 |
| 1.00 | 0.2132 | +0.0557 |
| 1.20 | 0.2202 | +0.0627 |
| 1.50 | 0.2286 | +0.0711 |
| 1.73 | 0.2315 | +0.0740 |
| 2.00 | 0.2342 | +0.0767 |
| 2.50 | 0.2383 | +0.0808 |

### structural (no production gamma — sweep applied raw)

| Gamma | Mean Brier | vs Market |
|-------|-----------|-----------|
| 0.50 | 0.2241 | +0.0666 |
| 0.60 | 0.2245 | +0.0670 |
| 0.70 | 0.2261 | +0.0686 |
| 0.80 | 0.2286 | +0.0711 |
| 0.90 | 0.2318 | +0.0743 |
| 1.00 | 0.2354 | +0.0779 |
| 1.20 | 0.2433 | +0.0858 |
| 1.50 | 0.2540 | +0.0965 |
| 1.73 | 0.2613 | +0.1038 |
| 2.00 | 0.2688 | +0.1113 |
| 2.50 | 0.2794 | +0.1219 |

### actor (production gamma=0.8)

| Gamma | Mean Brier | vs Market |
|-------|-----------|-----------|
| 0.50 | 0.2314 | +0.0739 |
| 0.60 | 0.2306 | +0.0731 |
| 0.70 | 0.2302 | +0.0727 |
| 0.80 | 0.2301 | +0.0726 |
| 0.90 | 0.2302 | +0.0727 |
| 1.00 | 0.2304 | +0.0729 |
| 1.20 | 0.2303 | +0.0728 |
| 1.50 | 0.2299 | +0.0724 |
| 1.73 | 0.2300 | +0.0725 |
| 2.00 | 0.2302 | +0.0727 |
| 2.50 | 0.2307 | +0.0732 |

### graphrag (production gamma=0.8)

| Gamma | Mean Brier | vs Market |
|-------|-----------|-----------|
| 0.50 | 0.2246 | +0.0671 |
| 0.60 | 0.2277 | +0.0702 |
| 0.70 | 0.2321 | +0.0746 |
| 0.80 | 0.2374 | +0.0799 |
| 0.90 | 0.2430 | +0.0855 |
| 1.00 | 0.2487 | +0.0912 |
| 1.20 | 0.2594 | +0.1019 |
| 1.50 | 0.2702 | +0.1127 |
| 1.73 | 0.2752 | +0.1177 |
| 2.00 | 0.2785 | +0.1210 |
| 2.50 | 0.2832 | +0.1257 |

### perspective (production gamma=0.8)

| Gamma | Mean Brier | vs Market |
|-------|-----------|-----------|
| 0.50 | 0.2429 | +0.0854 |
| 0.60 | 0.2463 | +0.0888 |
| 0.70 | 0.2506 | +0.0931 |
| 0.80 | 0.2555 | +0.0980 |
| 0.90 | 0.2608 | +0.1033 |
| 1.00 | 0.2663 | +0.1088 |
| 1.20 | 0.2769 | +0.1194 |
| 1.50 | 0.2922 | +0.1347 |
| 1.73 | 0.3023 | +0.1448 |
| 2.00 | 0.3123 | +0.1548 |
| 2.50 | 0.3263 | +0.1688 |

### debate (production gamma=0.8)

| Gamma | Mean Brier | vs Market |
|-------|-----------|-----------|
| 0.50 | 0.2446 | +0.0871 |
| 0.60 | 0.2495 | +0.0920 |
| 0.70 | 0.2553 | +0.0978 |
| 0.80 | 0.2617 | +0.1042 |
| 0.90 | 0.2684 | +0.1109 |
| 1.00 | 0.2752 | +0.1177 |
| 1.20 | 0.2874 | +0.1299 |
| 1.50 | 0.3030 | +0.1455 |
| 1.73 | 0.3133 | +0.1558 |
| 2.00 | 0.3232 | +0.1657 |
| 2.50 | 0.3360 | +0.1785 |


## Reproducing / Re-running

The benchmark dataset is a frozen fixture at `data/fixtures/kalshi_benchmark_midrange.json` (114 questions with ground-truth outcomes and market probabilities). After changing engine code, re-run engines against the same dataset:

```bash
python scripts/build_kalshi_fixture.py --engines-only
```

This skips market discovery and candlestick sync (~6 min), loading the saved fixture directly. Engine runs take ~1.5-3 hours depending on concurrency and LLM provider.

To rebuild the full dataset from scratch (e.g. with more markets or different cutoffs):

```bash
python scripts/build_kalshi_fixture.py              # full pipeline
python scripts/build_kalshi_fixture.py --skip-engines  # data only, no engine runs
python scripts/build_kalshi_fixture.py --discover-only  # just discover markets
```

After engine runs complete, generate this report and ingest into the forecast DB:

```bash
python scripts/analyze_kalshi_benchmark.py     # generate markdown report
python scripts/build_kalshi_fixture.py --ingest  # store in forecast DB
```

Ingested results appear on the `/benchmark` page tagged as **benchmark** (source type `kalshi_benchmark`). These are backtested forecasts — the engines were given questions from already-settled Kalshi markets and scored against known outcomes. They are not live Forward Look output.

**Note**: The fixture files (`data/fixtures/`) are not committed to git — they contain Kalshi market data specific to this instance. Other users would need to run the full pipeline first to build their own fixture, which requires a Kalshi API key.

## Caveats

1. **Hindsight bias**: LLM engines may "remember" outcomes of older markets from training data. The engine-vs-engine comparison is still fair (same LLM), but absolute Brier scores may be artificially low.
2. **Single snapshot**: No historical candlestick data available from Kalshi API for settled markets. Market probability is the last traded price (close to 0/1 for most settled markets).
3. **Extreme skew**: 90%+ of markets have extreme probabilities (≤0.05 or ≥0.95), making the mid-range subset the most meaningful comparison.
4. **Knowledge coverage**: Most Kalshi categories don't overlap with existing Nexus topics. Knowledge-augmented engines (GraphRAG, perspective) have limited context to work with.
