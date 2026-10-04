"""Data partition: external centre, patient-level folds and leave-one-centre-out.

The leakage-relevant unit is the patient (one index aneurysm per patient), nested in centres. The external centre is
removed first and never enters training, tuning or threshold selection. Within the development cohort, folds are
formed by patient before any preprocessing statistic, augmentation or model is fitted.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from ..types import CaseRecord


@dataclass
class Fold:
    name: str
    train: List[str]
    val: List[str]


@dataclass
class SplitManifest:
    """All partitions of one study, written to ``data/splits/`` (case IDs only, no images)."""

    development: List[str]
    external: List[str]
    external_centre: Optional[str]
    folds: List[Fold] = field(default_factory=list)
    loco: List[Fold] = field(default_factory=list)
    seed: int = 0

    def to_json(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "SplitManifest":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d["folds"] = [Fold(**f) for f in d.get("folds", [])]
        d["loco"] = [Fold(**f) for f in d.get("loco", [])]
        return cls(**d)

    def nnunet_splits(self) -> List[Dict[str, List[str]]]:
        """Folds in the ``splits_final.json`` format of nnU-Net v2."""
        return [{"train": f.train, "val": f.val} for f in self.folds]


def external_split(records: Sequence[CaseRecord], external_centre: str) -> tuple[List[CaseRecord], List[CaseRecord]]:
    dev = [r for r in records if r.centre != external_centre]
    ext = [r for r in records if r.centre == external_centre]
    if not ext:
        raise ValueError(f"no case from the external centre {external_centre!r}")
    assert_no_leakage({"development": dev, "external": ext})
    return dev, ext


def patient_kfold(records: Sequence[CaseRecord], k: int = 5, seed: int = 0, stratify_by: Optional[str] = "treatment") -> List[Fold]:
    """Patient-level k-fold split; patients are shuffled with ``seed`` and dealt round-robin within strata."""
    rng = np.random.default_rng(seed)
    by_patient: Dict[str, List[CaseRecord]] = {}
    for r in records:
        by_patient.setdefault(r.patient_id, []).append(r)
    strata: Dict[str, List[str]] = {}
    for pid, rs in by_patient.items():
        key = str(getattr(rs[0], stratify_by)) if stratify_by else "all"
        strata.setdefault(key, []).append(pid)
    fold_of: Dict[str, int] = {}
    offset = 0
    for key in sorted(strata):
        pids = sorted(strata[key])
        rng.shuffle(pids)
        for j, pid in enumerate(pids):
            fold_of[pid] = (j + offset) % k
        offset += len(pids)
    folds = []
    for i in range(k):
        val = sorted(r.case_id for r in records if fold_of[r.patient_id] == i)
        train = sorted(r.case_id for r in records if fold_of[r.patient_id] != i)
        folds.append(Fold(f"fold_{i}", train, val))
    _check_folds(records, folds)
    return folds


def leave_one_centre_out(records: Sequence[CaseRecord]) -> List[Fold]:
    centres = sorted({r.centre for r in records})
    if len(centres) < 2:
        raise ValueError("leave-one-centre-out needs at least two centres")
    folds = [Fold(f"loco_{c.replace(' ', '_')}", sorted(r.case_id for r in records if r.centre != c),
                  sorted(r.case_id for r in records if r.centre == c)) for c in centres]
    _check_folds(records, folds)
    return folds


def make_manifest(records: Sequence[CaseRecord], external_centre: str, k: int = 5, seed: int = 0) -> SplitManifest:
    dev, ext = external_split(records, external_centre)
    return SplitManifest(development=sorted(r.case_id for r in dev), external=sorted(r.case_id for r in ext),
                         external_centre=external_centre, folds=patient_kfold(dev, k, seed), loco=leave_one_centre_out(records),
                         seed=seed)


def assert_no_leakage(partitions: Dict[str, Iterable[CaseRecord]]) -> None:
    """Raise if a patient appears in more than one partition."""
    seen: Dict[str, str] = {}
    for name, rs in partitions.items():
        for r in rs:
            if r.patient_id in seen and seen[r.patient_id] != name:
                raise ValueError(f"patient {r.patient_id} is in both {seen[r.patient_id]!r} and {name!r}")
            seen[r.patient_id] = name


def _check_folds(records: Sequence[CaseRecord], folds: Sequence[Fold]) -> None:
    pid = {r.case_id: r.patient_id for r in records}
    val_seen: Dict[str, int] = {}
    for f in folds:
        tr, va = {pid[c] for c in f.train}, {pid[c] for c in f.val}
        if tr & va:
            raise ValueError(f"{f.name}: patients {sorted(tr & va)[:5]} in both train and validation")
        for c in f.val:
            val_seen[c] = val_seen.get(c, 0) + 1
    if folds and folds[0].name.startswith("fold_") and set(val_seen) != set(pid):
        raise ValueError("every case must be in exactly one validation fold")
    if any(v != 1 for v in val_seen.values()):
        raise ValueError("a case is in more than one validation fold")
