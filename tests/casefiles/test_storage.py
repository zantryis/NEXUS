"""Tests for casefile manifest loading and persistence."""

import json

import yaml

from nexus.casefiles.models import (
    CaseMetadata,
    CaseOverview,
    CaseReview,
    CasefileBundle,
)
from nexus.casefiles.storage import load_case_definition, save_casefile


def test_load_case_definition_without_seed(tmp_path):
    case_dir = tmp_path / "casefiles" / "sample"
    case_dir.mkdir(parents=True)
    (case_dir / "case.yaml").write_text(
        yaml.dump(
            {
                "slug": "sample",
                "title": "Sample Case",
                "question": "What happened?",
                "time_bounds": {"start": "2020-01-01", "end": "2020-01-02"},
                "hypothesis_seeds": ["Hypothesis A", "Hypothesis B", "Hypothesis C"],
                "reading_levels": ["short", "standard", "deep"],
            },
            sort_keys=False,
        )
    )

    loaded = load_case_definition(case_dir)
    assert loaded.case.slug == "sample"
    assert loaded.seeds is None
    assert loaded.bundle is None


def test_load_case_definition_with_seed_and_bundle(tmp_path):
    case_dir = tmp_path / "casefiles" / "sample"
    case_dir.mkdir(parents=True)
    (case_dir / "case.yaml").write_text(
        yaml.dump(
            {
                "slug": "sample",
                "title": "Sample Case",
                "question": "What happened?",
                "time_bounds": {"start": "2020-01-01", "end": "2020-01-02"},
                "hypothesis_seeds": ["Hypothesis A", "Hypothesis B", "Hypothesis C"],
                "reading_levels": ["short", "standard", "deep"],
            },
            sort_keys=False,
        )
    )
    (case_dir / "seed.yaml").write_text(
        yaml.dump(
            {
                "sources": [
                    {
                        "id": "src-1",
                        "label": "Primary source",
                        "url": "https://example.com/report.pdf",
                        "kind": "report",
                        "role": "primary",
                        "source_class": "official",
                        "priority": 9,
                    }
                ]
            },
            sort_keys=False,
        )
    )

    bundle = CasefileBundle(
        metadata=CaseMetadata(
            slug="sample",
            title="Sample Case",
            question="What happened?",
            generated_at="2026-03-21T00:00:00+00:00",
            last_updated="2026-03-21T00:00:00+00:00",
            presentable=False,
        ),
        overview=CaseOverview(
            best_current_account="Draft account",
            confidence_label="Low",
            reading_levels={
                "short": "Short",
                "standard": "Standard draft account that is long enough.",
                "deep": "Deep draft account.",
            },
        ),
        review=CaseReview(),
    )
    save_casefile(case_dir, bundle)

    loaded = load_case_definition(case_dir)
    assert loaded.seeds is not None
    assert loaded.seeds.sources[0].id == "src-1"
    assert loaded.bundle is not None
    assert loaded.bundle.metadata.slug == "sample"


def test_save_casefile_persists_valid_json(tmp_path):
    case_dir = tmp_path / "casefiles" / "sample"
    bundle = CasefileBundle(
        metadata=CaseMetadata(
            slug="sample",
            title="Sample Case",
            question="What happened?",
            generated_at="2026-03-21T00:00:00+00:00",
            last_updated="2026-03-21T00:00:00+00:00",
            presentable=True,
        ),
        overview=CaseOverview(
            best_current_account="Account",
            confidence_label="Moderate",
            reading_levels={
                "short": "Short",
                "standard": "Standard account that is comfortably longer than the short version.",
                "deep": "Deep account.",
            },
        ),
        review=CaseReview(presentable=True, verdict="presentable"),
    )
    path = save_casefile(case_dir, bundle)
    payload = json.loads(path.read_text())
    assert payload["metadata"]["slug"] == "sample"
    assert payload["review"]["verdict"] == "presentable"

