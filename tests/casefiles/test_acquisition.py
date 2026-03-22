"""Tests for casefile acquisition helpers."""

from types import SimpleNamespace

from nexus.casefiles.acquisition import (
    canonicalize_url,
    dedupe_candidates,
    extract_pdf_text,
    select_balanced_candidates,
)
from nexus.casefiles.models import CandidateDocument


def _candidate(
    candidate_id: str,
    *,
    url: str,
    source_class: str,
    priority: int = 5,
    role: str = "secondary",
    kind: str = "article",
    label: str | None = None,
):
    return CandidateDocument(
        id=candidate_id,
        label=label or candidate_id,
        title_hint=label or candidate_id,
        url=url,
        kind=kind,
        role=role,
        source_class=source_class,
        priority=priority,
    )


def test_dedupe_candidates_removes_duplicate_urls_and_titles():
    candidates = [
        _candidate("a", url="https://example.com/report?utm_source=x", source_class="official", priority=9),
        _candidate("b", url="https://example.com/report", source_class="official", priority=3),
        _candidate("c", url="https://other.example.com/story", source_class="media", label="Same Title"),
        _candidate("d", url="https://other.example.com/other-story", source_class="media", label="Same Title"),
    ]
    deduped = dedupe_candidates(candidates)
    assert [item.id for item in deduped] == ["a", "c"]


def test_select_balanced_candidates_preserves_source_class_spread():
    candidates = [
        _candidate("off", url="https://example.com/off", source_class="official", priority=10),
        _candidate("inv", url="https://example.com/inv", source_class="investigation", priority=8),
        _candidate("ana", url="https://example.com/ana", source_class="analysis", priority=7),
        _candidate("med", url="https://example.com/med", source_class="media", priority=6),
        _candidate("off-2", url="https://example.com/off-2", source_class="official", priority=5),
    ]
    selected = select_balanced_candidates(candidates, 4)
    assert {item.source_class for item in selected} == {"official", "investigation", "analysis", "media"}


def test_archive_url_is_accepted_without_canonical_damage():
    archive_url = "https://web.archive.org/web/20180814134451/http://mh370.mot.gov.my/MH370SafetyInvestigationReport.pdf"
    assert canonicalize_url(archive_url) == archive_url


def test_extract_pdf_text_success(monkeypatch):
    class FakePage:
        def extract_text(self):
            return "MH370 report text " * 40

    class FakeReader:
        def __init__(self, _):
            self.pages = [FakePage()]
            self.metadata = SimpleNamespace(title="MH370")

    monkeypatch.setattr("nexus.casefiles.acquisition.PdfReader", FakeReader)
    text, title, error = extract_pdf_text(b"%PDF-fake")
    assert "MH370 report text" in text
    assert title == "MH370"
    assert error is None


def test_extract_pdf_text_handles_scanned_or_unsupported_pdf(monkeypatch):
    class FakePage:
        def extract_text(self):
            return ""

    class FakeReader:
        def __init__(self, _):
            self.pages = [FakePage()]
            self.metadata = SimpleNamespace(title="Scanned")

    monkeypatch.setattr("nexus.casefiles.acquisition.PdfReader", FakeReader)
    text, title, error = extract_pdf_text(b"%PDF-fake")
    assert text == ""
    assert title == "Scanned"
    assert error == "pdf_text_too_thin"

