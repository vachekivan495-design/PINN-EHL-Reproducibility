from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence, TypeVar

from .data import CaseArtifact, CaseMetadata, assert_unique_conditions, canonical_condition_key


REQUIRED_SPLITS = ("train", "validation", "test_interpolation")
TCase = TypeVar("TCase", CaseArtifact, CaseMetadata)


@dataclass(frozen=True)
class SplitAssignment:
    groups: Mapping[str, tuple[str, ...]]

    @classmethod
    def load(cls, path: str | Path) -> "SplitAssignment":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        groups_raw = raw.get("splits", raw)
        if not isinstance(groups_raw, Mapping):
            raise ValueError("Split manifest must contain a mapping named 'splits'")
        groups = {
            str(name): tuple(str(case_id) for case_id in case_ids)
            for name, case_ids in groups_raw.items()
        }
        assignment = cls(groups=groups)
        assignment.validate_shape()
        return assignment

    def validate_shape(self) -> None:
        missing = [name for name in REQUIRED_SPLITS if not self.groups.get(name)]
        if missing:
            raise ValueError(f"Required non-empty splits are missing: {missing}")
        seen: dict[str, str] = {}
        for split_name, case_ids in self.groups.items():
            for case_id in case_ids:
                if case_id in seen:
                    raise ValueError(
                        f"Case ID {case_id!r} appears in {seen[case_id]!r} and {split_name!r}"
                    )
                seen[case_id] = split_name

    def bind(self, cases: Sequence[TCase]) -> dict[str, list[TCase]]:
        self.validate_shape()
        assert_unique_conditions(cases)
        by_id = {case.case_id: case for case in cases}
        requested = {case_id for ids in self.groups.values() for case_id in ids}
        missing = sorted(requested - set(by_id))
        extra = sorted(set(by_id) - requested)
        if missing or extra:
            raise ValueError(f"Split/case mismatch: missing={missing}, unassigned={extra}")
        bound = {
            split_name: [by_id[case_id] for case_id in case_ids]
            for split_name, case_ids in self.groups.items()
        }
        condition_owner: dict[tuple[float, ...], str] = {}
        for split_name, split_cases in bound.items():
            for case in split_cases:
                key = canonical_condition_key(case.condition)
                previous = condition_owner.get(key)
                if previous is not None and previous != split_name:
                    raise ValueError(
                        f"Physical condition for {case.case_id!r} crosses {previous!r} and {split_name!r}"
                    )
                condition_owner[key] = split_name
        return bound
