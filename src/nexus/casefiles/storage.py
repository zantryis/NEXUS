"""File-backed casefile loading and persistence helpers."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from nexus.casefiles.models import (
    CaseConfig,
    CasefileBundle,
    LoadedCaseDefinition,
    SeedManifest,
)


def casefiles_root(data_dir: Path) -> Path:
    """Return the casefiles root under the configured data directory."""
    return data_dir / "casefiles"


def case_dir(data_dir: Path, slug: str) -> Path:
    """Return the directory for one case slug."""
    return casefiles_root(data_dir) / slug


def load_case_definition(case_path: Path) -> LoadedCaseDefinition:
    """Load a case directory contract from disk."""
    case_file = case_path / "case.yaml"
    if not case_file.exists():
        raise FileNotFoundError(f"Missing case.yaml in {case_path}")

    case = CaseConfig.model_validate(yaml.safe_load(case_file.read_text()) or {})

    seed_file = case_path / "seed.yaml"
    seeds = None
    if seed_file.exists():
        seeds = SeedManifest.model_validate(yaml.safe_load(seed_file.read_text()) or {})

    bundle_file = case_path / "casefile.json"
    bundle = None
    if bundle_file.exists():
        bundle = CasefileBundle.model_validate(json.loads(bundle_file.read_text()))

    return LoadedCaseDefinition(
        path=str(case_path),
        case=case,
        seeds=seeds,
        bundle=bundle,
    )


def load_case(data_dir: Path, slug: str) -> LoadedCaseDefinition:
    """Load one case by slug."""
    return load_case_definition(case_dir(data_dir, slug))


def list_cases(data_dir: Path) -> list[LoadedCaseDefinition]:
    """List case definitions available under data/casefiles."""
    root = casefiles_root(data_dir)
    if not root.exists():
        return []

    cases: list[LoadedCaseDefinition] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        try:
            cases.append(load_case_definition(child))
        except FileNotFoundError:
            continue
    return cases


def save_casefile(case_path: Path, bundle: CasefileBundle) -> Path:
    """Persist the canonical casefile bundle."""
    case_path.mkdir(parents=True, exist_ok=True)
    bundle_path = case_path / "casefile.json"
    bundle_path.write_text(bundle.model_dump_json(indent=2, exclude_none=True) + "\n")
    return bundle_path
