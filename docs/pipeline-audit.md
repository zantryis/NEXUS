# Nexus Pipeline Audit

**Date**: 2026-03-14 (original), updated 2026-03-14 post-hardening
**Scope**: Full pipeline workflow, stage-by-stage quality assessment, evaluation system review
**Method**: Code-level review of all pipeline stages, prompts, data models, and benchmark outputs

---

## Pipeline Overview

The pipeline runs 10 stages per topic:

```
RSS Poll → Dedup → Ingest → Filter (2-pass) → Extract Events → Dedup Events
    → Resolve Entities → Synthesize → Render → Persist
```

Each stage is reviewed below with a verdict: **working**, **partially working**, or **broken**.

---

## Stage 1: Source Polling & Recency Filter

**Files**: `sources/polling.py`, `pipeline.py:111-121`
**Verdict**: Working

Polls RSS feeds via feedparser, drops articles older than 48 hours. Source metadata
(affiliation, country, tier, language) flows from the registry into each `ContentItem`.

**No issues found.** The recency filter is simple and correct. The 48-hour window is
appropriate for a daily pipeline.

---

## Stage 2: URL Dedup & Ingestion Cap

**Files**: `ingestion/dedup.py`, `pipeline.py:126-149`
**Verdict**: Working

Deduplicates by URL. Caps ingestion at 250 articles, prioritizing dated articles
(most recent first). Full text extracted via trafilatura with per-domain rate limiting
(2 concurrent per domain, 10 global).

**Minor concern**: No retry on HTTP failure during ingestion. An article that times out
is silently lost. In practice this rarely matters because the pipeline has redundant
coverage from multiple sources.

**Recommendation**: No action needed. Acceptable loss rate.

---

## Stage 3: Two-Pass Filtering

**Files**: `filtering/filter.py`
**Verdict**: Partially working

### Pass 1: Relevance (batch scoring, 10 articles per LLM call)

Scores articles 1-10 against topic definition. Threshold defaults to 5.0.
Text capped at 1000 chars. Source affiliation/country metadata included in prompt.

**Issue**: Pass 1 has no knowledge of existing events. It scores relevance to the
*topic definition*, not to *what's new*. An article restating known facts scores
the same as one breaking new ground. This is partially mitigated by Pass 2's
novelty check, but articles that are relevant-but-redundant still consume Pass 2
LLM budget.

### Pass 2: Significance + Novelty (5 articles per call, 2000-char text)

Scores significance (1-10) and novelty (boolean) against last 7 days of events.
Keeps articles where significance >= 4 OR is_novel == true.

**~~Issue~~ Fixed**: The `is_novel` fallback was too permissive. When Pass 2 batch parsing
failed, the fallback assumed all articles were novel and significance 5, promoting
everything through the filter. **Now fixed**: fallback returns `significance=3,
is_novel=False`, which causes items to be rejected by the `sig >= 4 or is_novel` gate.

### Composite Score

```python
composite = (relevance * 0.4 + significance * 0.6) * novelty_bonus
```

Where `novelty_bonus` = 1.0 if novel, 0.7 if not. This formula is reasonable.
The 0.6 weight on significance correctly prioritizes important articles over
merely relevant ones.

### Perspective Diversity Selection

For `high` diversity: guarantees 20% minimum per affiliation type present.
For `medium`: 10%. For `low`: pure score ranking.

**Issue**: Diversity selection runs AFTER significance filtering. If all
high-significance articles happen to come from one affiliation (plausible for
niche topics), the diversity constraint drops them in favor of lower-scoring
articles from other affiliations. This trades quality for balance.

The algorithm at line 232-255 guarantees `min_per_type` slots per affiliation,
then fills remaining slots by score. If there are 4 affiliation types and
`max_items=30` with `high` diversity, each type gets at least 6 slots
(30 * 0.2). If one type only has 2 articles, those 4 unused slots aren't
redistributed — they're filled from the `remaining` pool by score, which
could come from any affiliation. This is actually fine behavior.

**Recommendation**:
- ~~Fix the Pass 2 parse error fallback to be conservative (reject, not accept)~~ Done
- Consider adding event context to Pass 1 (costs ~20% more tokens, reduces
  redundant Pass 2 calls)

---

## Stage 4: Event Extraction

**Files**: `knowledge/events.py:120-214`
**Verdict**: Working (framing extraction + tone validation implemented)

For each filtered article, LLM extracts: date, summary, entities, significance,
editorial_tone (constrained vocabulary: 8 words), editorial_focus (5-10 words),
actor_framing (5-15 words). These are concatenated into a `framing` string per
source: `[tone] focus; actor_framing`.

**~~Issue 2~~ Fixed: Editorial tone validation.** `VALID_TONES` constant (8 words)
now validates `editorial_tone` after extraction. Invalid tones (e.g., "balanced",
"hawkish") are replaced with "neutral" and logged at DEBUG level. This ensures
downstream divergence analysis compares structured categories, not free-form text.

**Issue 1: Silent loss on parse failure.** (Open) Line 212-214:
```python
except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
    logger.warning(f"Failed to extract event from {item.url}: {e}")
    return None
```
If the LLM returns malformed JSON, the article is silently dropped. No retry,
no fallback to a simpler extraction. There's no tracking of how often this
happens.

**Issue 3: Entity extraction is string-only.** (Open) The LLM returns entity names
as raw strings. "Iran", "Islamic Republic of Iran", and "Tehran" could all
appear as separate entities for the same article. Entity resolution happens
later (Stage 6), but the structural dedup in Stage 5 depends on entity
overlap — and it uses these raw, unresolved strings.

**Recommendation**:
- Add a single retry on JSON parse failure before returning None
- ~~Validate editorial_tone against the 8-word vocabulary; default to "neutral"
  if invalid~~ Done
- Track extraction failure rate as a pipeline metric

---

## Stage 5: Structural Event Dedup

**Files**: `knowledge/events.py:52-96`, `pipeline.py:192-208`
**Verdict**: Partially working

Deduplicates extracted events by checking entity overlap (Jaccard similarity
on case-insensitive strings) and date proximity (same day or adjacent).
Threshold: 60% entity overlap. When duplicate, merges sources and keeps
higher significance.

**Issue**: Dedup uses raw entity strings, not resolved canonical names.
"Khamenei" and "Ali Khamenei" have 0% overlap. "US" and "United States"
have 0% overlap. This means near-duplicate events with differently-phrased
entities are NOT caught, resulting in redundant events entering synthesis.

The pipeline runs entity resolution AFTER dedup (pipeline.py:216-262).
This ordering means the dedup stage operates on unreliable data.

**Why it's this way**: Entity resolution requires an LLM call that depends
on the accumulated entity graph. You need events before you can resolve
entities. It's a chicken-and-egg problem.

**Recommendation**: This is a real architectural issue. Two options:
1. **Cheap fix**: Add a fuzzy string matching layer (Levenshtein distance)
   to `is_duplicate_event` alongside exact entity overlap. This catches
   "Khamenei" vs "Ali Khamenei" without an LLM call.
2. **Proper fix**: Run a lightweight first-pass entity normalization
   (rule-based: country name mapping, common abbreviation expansion)
   before dedup. Save LLM-based resolution for the canonical step.

---

## Stage 6: Entity Resolution

**Files**: `knowledge/entities.py`
**Verdict**: Working (with caveats)

LLM maps raw entity strings to canonical names against up to 200 known
entities from the store. Conservative merging ("only merge if CERTAIN").
Falls back to treating each raw name as its own entity on parse failure.

**Caveat**: No confidence scores. Every resolution is treated as equally
certain. An ambiguous merge (is "IAEA Director" the same as "Rafael Grossi"?)
looks identical to an obvious one ("US" → "United States"). There's no
mechanism to flag uncertain resolutions for review.

**Recommendation**: No action needed now. This works well enough for the
current pipeline. Confidence scores would matter if you build a user-facing
entity graph, but not yet.

---

## Stage 7: Knowledge Synthesis

**Files**: `synthesis/knowledge.py`
**Verdict**: Core logic works; convergence now validated; divergence under evaluation

This is the heart of the pipeline. Events + articles + background summaries
go in; `TopicSynthesis` with `NarrativeThread` objects comes out.

### What works

- **Thread grouping**: The LLM groups events into narrative threads with
  headlines, entity lists, and significance ratings. The scope-aware prompt
  (broad vs narrow topics) helps the LLM calibrate thread granularity.
- **Event formatting**: Events are formatted with full source metadata and
  per-source framing strings, giving the LLM rich input for analysis.
- **Article snippets**: For multi-source events, 200-char article excerpts
  are appended to give the LLM raw text to compare framing.

### Convergence: now validated programmatically

~~The word "ideally" makes this a suggestion, not a rule.~~

**Implemented**: Source independence is now a deterministic function in
`events.py`: `are_independent(source_a, source_b)` returns True if sources
differ on `affiliation` OR `country`. Missing metadata → assume independent
(benefit of doubt).

**Post-synthesis validation** (`_validate_convergence()` in `knowledge.py`):
after the LLM generates convergence entries, each entry's `confirmed_by`
outlets are checked for at least one independent pair. Entries where all
confirming outlets share the same affiliation AND country are stripped.
Single-source convergence entries are also removed. This runs automatically
on every synthesis.

### Divergence: structural fix implemented, pending re-evaluation

Suite H pre-implementation baseline (4 topics, all 4 prompt variants):

| Variant | divergence_detection score |
|---------|---------------------------|
| baseline | 2.3 +/- 0.5 |
| broadened | 2.0 +/- 0.0 |
| structured | 2.0 +/- 0.0 |
| encouraged | 2.0 +/- 0.0 |

**Diagnosed root cause chain**: The LLM lacked structured framing data to
compare across sources. Without explicit per-source `[tone] focus; actors`
annotations, the synthesis LLM had no basis for identifying genuine framing
differences.

**Structural fix implemented** (schema v9):
1. Extraction prompt now requests `editorial_tone`, `editorial_focus`,
   `actor_framing` per source
2. `editorial_tone` validated against 8-word vocabulary (`VALID_TONES`)
3. Framing concatenated as `[tone] focus; actors` in source dicts
4. Synthesis prompt receives per-source framing + article snippets
5. Default divergence instructions reference structured framing data

**Status**: Suite H needs re-running post-implementation to measure
improvement. The structural data is now flowing; the question is whether
the synthesis LLM uses it to generate genuine divergence entries.

### Thread Consolidation: now monitored

~~The phrase "same causal chain" is vague and unenforceable.~~

**Implemented**: `_check_thread_overlaps()` in `knowledge.py` computes
Jaccard similarity on `key_entities` across all thread pairs after synthesis.
Pairs with overlap > 0.5 are logged as warnings. This provides observability
for a known LLM consolidation weakness without auto-merging (which risks
incorrectly combining threads about different aspects of the same entities).

---

## Stage 8: Thread Persistence

**Files**: `synthesis/threads.py`, `knowledge/store.py`
**Verdict**: Working

Two-stage matching: entity overlap (Jaccard) for auto-match (>= 0.5) or
LLM confirmation (0.3-0.5). Below 0.3 = new thread. Lifecycle:
emerging → active (2+ days of events) → stale → resolved.

**~~Minor issue~~ Fixed**: Thread staleness now implemented.
`check_staleness()` in `threads.py` demotes active/emerging threads to
"stale" after `STALE_AFTER_DAYS` (14) days without events. Resolved
threads are never demoted. `KnowledgeStore.mark_stale_threads()` performs
bulk staleness updates via SQL, checking the latest linked event date for
each thread.

---

## Stage 9: Briefing Rendering

**Files**: `synthesis/renderers.py`
**Verdict**: Working

Renders `TopicSynthesis` into markdown briefing via LLM. Style variants
(analytical, conversational, editorial). Benchmark text quality scores
range 7-9 across styles, with analytical scoring highest.

**No issues found.** This stage works well.

---

## Stage 10: Metrics & Persistence

**Files**: `evaluation/metrics.py`, `pipeline.py:389-398`
**Verdict**: Working

Automated metrics (Shannon entropy, convergence ratio, language coverage,
dedup ratio, independent convergence ratio) computed on every run. No LLM
needed. Saved to YAML.

**~~Issue~~ Fixed**: `independent_convergence_ratio()` now supplements
`convergence_ratio`. It uses `has_independent_sources()` from `events.py`
to count only events where at least one source pair differs on affiliation
OR country. Both metrics are included in `compute_run_metrics()` output —
the original `convergence_ratio` is kept for backward compatibility.

---

## Evaluation System Review

### What exists

| Component | File | What it does |
|-----------|------|-------------|
| Automated metrics | `evaluation/metrics.py` (103 lines) | Shannon entropy, convergence ratio, language coverage — no LLM |
| LLM-as-judge | `evaluation/judge.py` (152 lines) | 5-dimension rubric (completeness, source_balance, convergence_accuracy, divergence_detection, entity_coverage) |
| Benchmark | `evaluation/benchmark.py` (522 lines) | Full pipeline vs naive baseline, style comparison |
| Experiment suites | `evaluation/experiment.py` (1132 lines) | 8 suites: A (pipeline vs baselines), B (filter threshold), C (diversity), D (style), E (cross-judge), F (scoring weights), G (model matrix), H (divergence variants) |

### What's wrong with the evaluation

**~~1. The naive baseline is a straw man.~~ Fixed.**

`build_naive_synthesis` now takes real extracted `Event` objects (with dates,
entities, summaries, source metadata) and creates one thread per event — no
grouping, no convergence, no divergence. This isolates the value added by
the synthesis LLM. Callers in `benchmark.py` and `experiment.py` updated.

**2. All quality signals are self-referential.**

The pipeline uses Gemini Flash for filtering, extraction, and synthesis.
The judge uses the agent model (Gemini Pro or same Flash) to evaluate.
There's no human validation anywhere. LLM judges LLM output, and you
report the LLM's opinion as the score.

The Suite E cross-judge experiment (Gemini vs DeepSeek) is a step in the
right direction, but inter-rater agreement between two LLMs doesn't prove
either is correct — it proves they share biases.

**3. Divergence detection is under evaluation.**

Suite H pre-implementation baseline was 2.0-2.3. The structural fix
(per-source framing extraction + tone validation + synthesis prompt updates)
is now implemented. Suite H needs re-running to measure improvement. If
scores remain at floor, human inspection of synthesis JSON output is the
next diagnostic step (are `divergence` arrays populated? are entries
substantive?).

**4. Seven experiment suites before core validation.**

Suites B (filter threshold), F (scoring weights), and G (model matrix) are
parameter optimization — tuning dials on a system that hasn't been validated
against reality. Suite C (diversity) and Suite D (style) measure secondary
concerns. Only Suite A (pipeline vs baseline) and Suite H (divergence) test
core claims, and both have the problems described above.

**5. The experiment is expensive and non-reproducible.**

The full 7-suite experiment cost $2.38 and took 7.3 hours (March 13 run).
Articles change daily, so results aren't reproducible. You can't re-run
the same experiment and compare — the corpus is different each time. The
article caching within a single run is good, but there's no frozen corpus
for cross-run comparison.

---

## Recommendations: What to Do

### Delete or Shelve

| Item | Action | Status |
|------|--------|--------|
| Suites B, C, D, F | Shelve (don't run until core is validated) | Still shelved |
| Suite G (model matrix) | Shelve until you have Claude/GPT access | Still shelved |
| ~~Naive baseline~~ | ~~Replace~~ | **Done** — uses real events, one thread per event |
| ~~`convergence_ratio` metric~~ | ~~Replace~~ | **Done** — `independent_convergence_ratio` added alongside original |
| Narrative pages (pages.py) | Shelve | Still shelved |

### Fix Programmatically (No Human Required)

| Fix | Status |
|-----|--------|
| ~~Define source independence deterministically~~ | **Done** — `are_independent()` in `events.py` |
| ~~Validate editorial_tone vocabulary~~ | **Done** — `VALID_TONES` + validation in `events.py` |
| ~~Fix Pass 2 parse error fallback~~ | **Done** — rejects on parse error (`sig=3, novel=False`) |
| **Add extraction failure rate metric** | Open — observability improvement |
| ~~Add thread staleness check~~ | **Done** — `check_staleness()` + `mark_stale_threads()` |
| ~~Replace naive baseline~~ | **Done** — real events, no grouping/convergence/divergence |
| ~~Add independent_convergence_ratio~~ | **Done** — in `metrics.py` + `compute_run_metrics()` |
| ~~Post-synthesis convergence validation~~ | **Done** — `_validate_convergence()` in `knowledge.py` |
| ~~Post-synthesis thread dedup check~~ | **Done** — `_check_thread_overlaps()` in `knowledge.py` |
| **Fuzzy entity matching in event dedup** | Deferred — risk of false merges with same-surname entities (e.g., "Khamenei" matching both Ali and Mojtaba Khamenei) |

### Requires Human Investigation

| Task | What to do | Expected time | Why a human is needed |
|------|-----------|---------------|----------------------|
| **Validate divergence post-fix** | Re-run Suite H. If scores improved, inspect a few entries for quality. If still at floor, inspect synthesis JSON to check if `divergence` arrays are populated and entries are substantive. | 1 hour | Only a human can judge whether a framing difference is "genuine." |
| **Build gold-standard dataset** | For 1 topic on 3 different days: read 10-15 source articles, write down the real threads, real convergences, real divergences. Save as structured JSON. Use as ground truth for future evaluation. | 2-3 hours | This is editorial judgment. No LLM can provide ground truth for editorial quality. |
| **Spot-check convergence claims** | Take 10 convergence entries from recent syntheses. Verify: did outlets actually report this fact? Convergence validation now strips non-independent entries, but factual accuracy still needs human spot-checks. | 1 hour | Verifying factual claims against source material requires reading articles. |
| **Spot-check entity resolution** | Pull 30 entity resolution mappings from the store. Check: were merges correct? Were there obvious misses (same entity, different name, not merged)? | 30 minutes | Judging entity identity requires domain knowledge about the topic. |

### Decided

| Decision | Resolution |
|----------|------------|
| **What counts as "independent"?** | Option (c): Different affiliation OR country. Implemented in `are_independent()`. Missing metadata → assume independent (benefit of doubt). |
| **What's the right naive baseline?** | Real extracted events, one thread per event, no synthesis. Implemented in `build_naive_synthesis()`. |
| **Should diversity selection run before or after significance filtering?** | After (current). No change needed — behavior is reasonable. |

---

## Priority Order (Updated)

1. ~~Define source independence + post-synthesis validation~~ **Done**
2. ~~Replace naive baseline~~ **Done**
3. ~~Fix Pass 2 fallback~~ **Done**
4. ~~Validate editorial tone vocabulary~~ **Done**
5. ~~Add thread staleness~~ **Done**
6. ~~Post-synthesis thread overlap detection~~ **Done**
7. ~~Add independent convergence ratio~~ **Done**

### Remaining

1. **Re-run Suite H** to validate divergence improvement post-fix
2. **Re-run Suite A** to measure synthesis value-add with credible baseline
3. **Build gold-standard dataset for 1 topic** (human, 2-3 hours)
4. **Add extraction failure rate metric** (code, small)
5. **Resume experiment suites B-G** after Suite H shows improvement

---

## Summary

The pipeline architecture is sound. The data model (events with per-source
framing, entity graph, persistent threads) is well-designed for the task.
The evaluation infrastructure is extensive but disconnected from ground
truth.

**Hardening completed (2026-03-14)**:
- **Convergence** is now deterministic — `are_independent()` defines source
  independence mechanically, `_validate_convergence()` strips non-independent
  entries post-synthesis
- **Framing extraction** validated — `editorial_tone` checked against 8-word
  vocabulary, invalid tones default to "neutral"
- **Filter quality** improved — Pass 2 parse errors now reject instead of
  accepting all articles
- **Thread lifecycle** complete — staleness demotion after 14 days with no events
- **Naive baseline** credible — uses real events, isolates synthesis value-add
- **Thread overlap** monitored — Jaccard > 0.5 flagged as warnings
- **Independent convergence** metric added alongside raw convergence ratio

**Remaining critical gap**:
- **Divergence detection** has structural fix implemented (per-source framing
  data + synthesis prompt updates) but is pending re-evaluation via Suite H.
  Pre-fix baseline was 2.0-2.3/10. If post-fix scores don't improve, human
  inspection of synthesis JSON output is needed.

Everything else — filter tuning, model selection, style comparison — is
second-order optimization that should wait until divergence is validated.
