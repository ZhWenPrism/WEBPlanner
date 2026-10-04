"""Augmentation, patch sampling and nnU-Net export."""
from __future__ import annotations

import importlib.util
import json
from functools import lru_cache

import numpy as np
import pytest

from webplanner.data import transforms as T
from webplanner.data.dataset import InfiniteSampler, PatchDataset, crop_with_padding
from webplanner.data.synthetic import make_phantom
from webplanner.types import LABELS, Volume

LABEL_SET = set(LABELS)
HAS_NIFTI = any(importlib.util.find_spec(m) is not None for m in ("SimpleITK", "nibabel"))


@lru_cache(maxsize=None)
def _phantom(seed: int = 0):
    return make_phantom(shape=(48, 48, 48), seed=seed)


def _pair():
    ph = _phantom()
    return ph.image.array.copy(), ph.labels.array.copy()


TRANSFORMS = [
    T.RandomRotation(p=1.0),
    T.RandomScaling(p=1.0),
    T.ElasticDeformation(alpha=(3.0, 3.0), sigma=(4.0, 4.0), p=1.0),
    T.SpatialTransform(rotation_deg=[(-20, 20), (0, 0), (-10, 10)], elastic=True, p_rotation=1.0, p_scale=1.0,
                       p_elastic=1.0, elastic_alpha=(2.0, 2.0), elastic_sigma=(4.0, 4.0)),
    T.GaussianNoise(p=1.0),
    T.GaussianBlur(p=1.0),
    T.BrightnessContrast(p_brightness=1.0, p_contrast=1.0),
    T.Gamma(p=1.0),
    T.Gamma(invert=True, p=1.0),
    T.Mirror(p_axis=1.0),
    T.default_pipeline(),
]


@pytest.mark.parametrize("transform", TRANSFORMS, ids=lambda t: type(t).__name__)
def test_transforms_keep_shape_dtype_and_label_set(transform):
    img, lab = _pair()
    out_img, out_lab = transform(img, lab, np.random.default_rng(0))
    assert out_img.shape == img.shape and out_lab.shape == lab.shape
    assert out_img.dtype == np.float32 and out_lab.dtype == lab.dtype
    assert set(np.unique(out_lab).tolist()) <= LABEL_SET
    assert np.isfinite(out_img).all()


@pytest.mark.parametrize("transform", [T.GaussianNoise(p=1.0), T.GaussianBlur(p=1.0), T.Gamma(p=1.0),
                                       T.BrightnessContrast(p_brightness=1.0, p_contrast=1.0)])
def test_intensity_transforms_leave_label_untouched(transform):
    img, lab = _pair()
    out_img, out_lab = transform(img, lab, np.random.default_rng(1))
    np.testing.assert_array_equal(out_lab, lab)
    assert not np.allclose(out_img, img)


def test_rotation_by_90_degrees_matches_rot90():
    lab = np.zeros((16, 16, 16), np.uint8)
    lab[3:7, 2:12, 5:9] = 1
    lab[9:13, 4:6, 1:15] = 3
    img = lab.astype(np.float32)
    rot = T.RandomRotation(rotation_deg=[(90, 90), (0, 0), (0, 0)], p=1.0)
    _, out = rot(img, lab, np.random.default_rng(0))
    assert any(np.array_equal(out, np.rot90(lab, k, axes=(1, 2))) for k in (1, -1))


def test_mirror_flips_image_and_label_together():
    img, lab = _pair()
    out_img, out_lab = T.Mirror(axes=(0,), p_axis=1.0)(img, lab, np.random.default_rng(0))
    np.testing.assert_array_equal(out_img, img[::-1])
    np.testing.assert_array_equal(out_lab, lab[::-1])


def test_spatial_transform_crops_to_patch_and_supports_channels():
    img, lab = _pair()
    st = T.SpatialTransform(patch_size=(24, 24, 24), p_rotation=1.0, p_scale=1.0)
    out_img, out_lab = st(np.stack([img, -img]), lab, np.random.default_rng(3))
    assert out_img.shape == (2, 24, 24, 24) and out_lab.shape == (24, 24, 24)
    centre = T.SpatialTransform(patch_size=(24, 24, 24), p_rotation=0.0, p_scale=0.0)
    _, cropped = centre(img, lab, np.random.default_rng(0))
    np.testing.assert_array_equal(cropped, lab[12:36, 12:36, 12:36])


def test_pipeline_is_reproducible_from_seed():
    img, lab = _pair()
    pipe = T.default_pipeline(elastic=True)
    a = pipe(img, lab, np.random.default_rng(42))
    b = pipe(img, lab, np.random.default_rng(42))
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_enlarged_size():
    assert T.enlarged_size((32, 32, 32), rotation_deg=0.0, scale_range=(1.0, 1.0)) == (32, 32, 32)
    big = T.enlarged_size((32, 32, 32))
    assert all(b > 32 for b in big)


def test_crop_with_padding():
    arr = np.arange(27).reshape(3, 3, 3)
    out = crop_with_padding(arr, (-1, 0, 1), (3, 3, 3), pad_value=-5)
    assert out.shape == (3, 3, 3)
    assert (out[0] == -5).all() and (out[:, :, 2] == -5).all()
    np.testing.assert_array_equal(out[1:, :, :2], arr[:2, :, 1:])


def test_patch_dataset_shapes_dtypes_and_oversampling():
    img, lab = _pair()
    ds = PatchDataset([(img, lab), (img, lab)], patch_size=(32, 24, 32), foreground_oversample=1.0, samples_per_epoch=6)
    assert len(ds) == 6 and ds.n_cases == 2
    for i in range(6):
        x, y = ds.sample(i)
        assert x.shape == (32, 24, 32) and y.shape == (32, 24, 32)
        assert x.dtype == np.float32 and y.dtype == np.int64
        assert (y > 0).any()
    np.testing.assert_array_equal(ds.sample(3)[0], ds.sample(3)[0])


def test_patch_dataset_pads_small_volumes_and_reads_npz(tmp_path):
    from webplanner.utils.io import save_volume

    img, lab = _pair()
    small_img, small_lab = img[14:34, 14:34, 14:34], lab[14:34, 14:34, 14:34]
    pi = save_volume(Volume(small_img, (0.3, 0.3, 0.3)), tmp_path / "img.npz")
    pl = save_volume(Volume(small_lab, (0.3, 0.3, 0.3)), tmp_path / "lab.npz")
    ds = PatchDataset([(pi, pl)], patch_size=32, foreground_oversample=0.5)
    x, y = ds.sample(0)
    assert x.shape == (32, 32, 32) and y.shape == (32, 32, 32)
    assert (y[:6] == 0).all() and np.isclose(x[0, 0, 0], small_img.min())
    assert int((y > 0).sum()) == int((small_lab > 0).sum())


def test_patch_dataset_with_augmentation_and_enlarged_region():
    img, lab = _pair()
    patch = (32, 32, 32)
    ds = PatchDataset([(img, lab)], patch, transform=T.default_pipeline(patch), sample_size=T.enlarged_size(patch))
    for i in range(4):
        x, y = ds.sample(i)
        assert x.shape == patch and set(np.unique(y).tolist()) <= LABEL_SET


def test_infinite_sampler_rounds_and_resume():
    it = iter(InfiniteSampler(4, seed=1))
    draws = [next(it) for _ in range(12)]
    assert len(set(draws)) == 12
    for r in range(3):
        assert sorted(d % 4 for d in draws[4 * r:4 * r + 4]) == [0, 1, 2, 3]
    it2 = iter(InfiniteSampler(4, seed=1, start=6))
    assert [next(it2) for _ in range(6)] == draws[6:]


def test_patch_dataset_torch_tensors():
    torch = pytest.importorskip("torch")
    from webplanner.data.dataset import infinite_loader

    img, lab = _pair()
    ds = PatchDataset([(img, lab), (img, lab)], patch_size=32)
    x, y = ds[0]
    assert x.shape == (1, 32, 32, 32) and x.dtype == torch.float32
    assert y.shape == (32, 32, 32) and y.dtype == torch.int64
    xb, yb = next(infinite_loader(ds, batch_size=2))
    assert xb.shape == (2, 1, 32, 32, 32) and yb.shape == (2, 32, 32, 32)


@pytest.mark.skipif(not HAS_NIFTI, reason="needs SimpleITK or nibabel")
def test_export_nnunet_dataset(tmp_path):
    from webplanner.data.nnunet_export import export_nnunet_dataset
    from webplanner.utils.io import load_volume

    cases = []
    for i in range(3):
        ph = _phantom(i)
        cases.append((f"syn{i:03d}", ph.image, ph.labels))
    folds = [{"train": ["syn000", "syn001"], "val": ["syn002"]}, {"train": ["syn001", "syn002"], "val": ["syn000"]}]
    root = export_nnunet_dataset(cases, tmp_path / "raw", dataset_id=501, folds=folds,
                                 preprocessed_dir=tmp_path / "preprocessed")
    assert root.name == "Dataset501_WEBPlanner"
    for cid, _, _ in cases:
        assert (root / "imagesTr" / f"{cid}_0000.nii.gz").exists()
        assert (root / "labelsTr" / f"{cid}.nii.gz").exists()
    meta = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
    assert meta["channel_names"] == {"0": "3DRA"}
    assert meta["labels"] == {"background": 0, "aneurysm_sac": 1, "parent_artery": 2, "branch_arteries": 3,
                              "neck_landing_zone": 4}
    assert meta["numTraining"] == 3 and meta["file_ending"] == ".nii.gz"
    splits = json.loads((root / "splits_final.json").read_text(encoding="utf-8"))
    assert splits == folds
    assert (tmp_path / "preprocessed" / root.name / "splits_final.json").exists()
    back = load_volume(root / "labelsTr" / "syn000.nii.gz")
    np.testing.assert_array_equal(back.array, cases[0][2].array)
    np.testing.assert_allclose(back.spacing, cases[0][2].spacing, rtol=1e-6)
    with pytest.raises(ValueError):
        export_nnunet_dataset(cases, tmp_path / "raw2", folds=[{"train": ["syn000"], "val": ["syn000"]}])
