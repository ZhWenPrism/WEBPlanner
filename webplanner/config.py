"""Experiment configuration: typed sections loaded from YAML (``configs/``), with command-line overrides.

Example::

    cfg = load_config("configs/experiment/development.yaml", overrides=["ranking.n_splits=10"])
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml


@dataclass
class DataConfig:
    manifest: str = "data/cohort.csv"            # one row per patient: case_id, patient_id, centre, treatment, ...
    image_dir: str = "data/processed/images"
    label_dir: str = "data/processed/labels"
    split_file: str = "data/splits/manifest.json"
    external_centre: str = "Centre 3"
    n_folds: int = 5
    spacing_mm: float = 0.30
    crop_mm: float = 40.0
    seed: int = 2026


@dataclass
class SegmentationConfig:
    backend: str = "nnunet"                     # "nnunet" (nnU-Net v2 3d_fullres) | "native" (webplanner trainer)
    network: str = "residual_unet"              # native backend: residual_unet | unet3d | swin_unetr
    comparators: List[str] = field(default_factory=lambda: ["unet3d", "swin_unetr"])
    nnunet_dataset_id: int = 501
    patch_size: List[int] = field(default_factory=lambda: [128, 128, 128])
    batch_size: int = 2
    epochs: int = 1000
    iterations_per_epoch: int = 250
    learning_rate: float = 1e-2
    weight_decay: float = 3e-5
    deep_supervision: bool = True
    mirror_tta: bool = True
    ensemble_folds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])


@dataclass
class GeometryConfig:
    origin_radius_mm: float = 1.5
    irregular_solidity: float = 0.90
    significant_angulation_deg: float = 60.0
    asymmetry_deg: float = 45.0


@dataclass
class DeploymentConfigSection:
    catalogue: str = "configs/device/web17_sl.yaml"
    elongation_k: float = 0.45                  # refitted on development patients with post-deployment VasoCT
    n_angles: int = 96
    n_levels: int = 60


@dataclass
class RankingConfig:
    l2_grid: List[float] = field(default_factory=lambda: [1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
    n_splits: int = 5
    selection_metric: str = "log_loss"
    top_k: int = 3
    review_quantile: float = 0.20
    image_score_backend: str = "descriptor"     # descriptor | cnn


@dataclass
class EvaluationConfig:
    n_bootstrap: int = 4000
    alpha: float = 0.05
    prespecified_subgroups: List[str] = field(default_factory=lambda: ["simple_case"])
    exploratory_subgroups: List[str] = field(default_factory=lambda: ["location", "ruptured", "device_width", "angulated", "irregular"])


@dataclass
class Config:
    experiment: str = "development"
    output_dir: str = "outputs"
    seed: int = 2026
    data: DataConfig = field(default_factory=DataConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    deployment: DeploymentConfigSection = field(default_factory=DeploymentConfigSection)
    ranking: RankingConfig = field(default_factory=RankingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
        return path


def _merge(dc: Any, values: Dict[str, Any]) -> Any:
    for f in fields(dc):
        if f.name not in values:
            continue
        cur, new = getattr(dc, f.name), values[f.name]
        if is_dataclass(cur) and isinstance(new, dict):
            _merge(cur, new)
        else:
            setattr(dc, f.name, new)
    unknown = set(values) - {f.name for f in fields(dc)}
    if unknown:
        raise KeyError(f"unknown configuration keys in {type(dc).__name__}: {sorted(unknown)}")
    return dc


def _load_yaml_with_bases(path: Path) -> Dict[str, Any]:
    d = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    bases = d.pop("defaults", [])
    merged: Dict[str, Any] = {}
    for b in bases:
        merged = _deep_update(merged, _load_yaml_with_bases((path.parent / b).resolve()))
    return _deep_update(merged, d)


def _deep_update(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def parse_overrides(overrides: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in overrides:
        key, _, raw = item.partition("=")
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def load_config(path: Optional[Union[str, Path]] = None, overrides: Sequence[str] = ()) -> Config:
    """Defaults, then the YAML file (with its ``defaults:`` chain), then ``key.sub=value`` overrides."""
    values: Dict[str, Any] = _load_yaml_with_bases(Path(path)) if path else {}
    values = _deep_update(values, parse_overrides(overrides))
    return _merge(Config(), values)
