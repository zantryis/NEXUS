"""Lightweight checks for the release-facing docs surface."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _read(rel_path: str) -> str:
    return (ROOT / rel_path).read_text()


def test_readme_leads_with_forward_look_story():
    readme = _read("README.md")
    assert "Forward Look" in readme
    assert "6 competing forecast engines" not in readme
    assert "forecast benchmark" not in readme


def test_static_landing_page_matches_release_scope():
    landing = _read("docs/index.html")
    assert "Forward Look" in landing
    assert "6 competing forecast engines" not in landing
    assert "python -m nexus serve" not in landing


def test_pipeline_page_presents_forward_look_as_public_story():
    pipeline = _read("docs/pipeline.html")
    assert "Forward Look" in pipeline
    assert "6 forecast engines" not in pipeline
    assert "6 competing forecast engines" not in pipeline


def test_benchmark_doc_is_lab_facing_and_not_linked_to_predictions_dashboard():
    benchmark = _read("docs/benchmark-results.md")
    assert "/predictions" not in benchmark
    assert "/benchmark" in benchmark
    assert "lab" in benchmark.lower() or "internal" in benchmark.lower()


def test_system_map_readme_marks_legacy_surface():
    system_map = _read("docs/system-map/README.md")
    assert "legacy" in system_map.lower()
    assert "docs/pipeline.html" in system_map
