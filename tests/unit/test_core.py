import numpy as np
import pytest

from webplanner.config import load_config, parse_overrides
from webplanner.data.preprocessing import Preprocessor, crop_cube, resample_isotropic, uncrop
from webplanner.data.splits import SplitManifest, assert_no_leakage, leave_one_centre_out, make_manifest, patient_kfold
from webplanner.data.synthetic import make_phantom, operator_choice, random_spec, synthetic_cases
from webplanner.device.catalogue import Catalogue, lookup
from webplanner.device.sizing_chart import chart_size
from webplanner.types import NECK, SAC, CaseRecord, DeviceSize, Volume
from webplanner.utils.io import load_volume, read_json, save_volume, write_json


# ------------------------------------------------------------------ catalogue and chart
def test_catalogue_has_29_sizes_in_range():
    cat = Catalogue.default()
    assert len(cat) == 29
    assert cat.widths.min() == 4.0 and cat.widths.max() == 11.0
    hs = [s.height for s in cat.sizes]
    assert min(hs) == 2.0 and max(hs) == 9.0


def test_catalogue_yaml_round_trip(tmp_path):
    import yaml

    cat = Catalogue.default()
    p = tmp_path / "cat.yaml"
    p.write_text(yaml.safe_dump(cat.to_dict()), encoding="utf-8")
    assert Catalogue.from_yaml(p).sizes == cat.sizes


def test_lookup_bounds_and_flags():
    bins = Catalogue.default().width_bins
    assert lookup(bins, 2.0) == (4.0, "below table")
    assert lookup(bins, 12.0) == (11.0, "above table")
    assert lookup(bins, 6.2)[0] == 8.0


def test_chart_upsizes_to_neck_and_uses_available_height():
    d = chart_size(mean_width=4.0, smallest_height=6.5, max_neck=6.6)
    assert d.size.width >= 6.6 and "upsized to the neck" in d.flags
    assert Catalogue.default().contains(d.size)


def test_within_one_size_and_acceptable():
    cat = Catalogue.default()
    a, b = DeviceSize(7.0, 4.0), DeviceSize(8.0, 5.0)
    assert cat.within_one_size(a, b)
    assert cat.clinically_acceptable(b, a) and not cat.clinically_acceptable(a, b)
    assert not cat.within_one_size(DeviceSize(5.0, 3.0), DeviceSize(7.0, 3.0))


# ------------------------------------------------------------------ phantoms
def test_phantom_labels_and_truth():
    ph = make_phantom()
    counts = np.bincount(ph.labels.array.ravel(), minlength=5)
    assert all(counts[1:] > 0)
    assert ph.truth["neck_width"] < ph.truth["mean_width"]
    # the neck landing zone lies at the base of the sac
    idx = np.argwhere(ph.labels.array == NECK) * ph.spec.spacing
    t = (idx - ph.truth["neck_center"]) @ ph.truth["normal"]
    assert t.max() < 1.0 and t.min() > -0.2


def test_random_specs_are_valid_and_reproducible():
    a = random_spec(np.random.default_rng(3))
    b = random_spec(np.random.default_rng(3))
    assert a == b
    ph = make_phantom(a)
    assert (ph.labels.array == SAC).sum() > 100


def test_operator_choice_in_catalogue():
    cat = Catalogue.default()
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = operator_choice(rng.uniform(3, 10), rng.uniform(3, 9), rng.uniform(2, 6), bool(rng.random() < 0.3), rng.uniform(0, 90), cat, rng)
        assert cat.contains(s)


# ------------------------------------------------------------------ splits
def _records(n=60):
    return [rec for rec, _ in synthetic_cases(n, seed=1)]


def test_patient_kfold_partitions_every_case_once():
    recs = _records()
    folds = patient_kfold(recs, k=5, seed=0)
    val = [c for f in folds for c in f.val]
    assert sorted(val) == sorted(r.case_id for r in recs)
    for f in folds:
        assert not set(f.train) & set(f.val)


def test_kfold_keeps_patients_together():
    recs = _records(20)
    recs += [CaseRecord(case_id="dup", patient_id=recs[0].patient_id, centre=recs[0].centre)]
    folds = patient_kfold(recs, k=4, seed=0)
    for f in folds:
        assert ("dup" in f.val) == (recs[0].case_id in f.val)


def test_manifest_external_never_in_development(tmp_path):
    recs = _records()
    m = make_manifest(recs, external_centre="Centre 3", k=5, seed=0)
    assert not set(m.development) & set(m.external)
    assert all(not set(f.train) & set(m.external) for f in m.folds)
    p = m.to_json(tmp_path / "manifest.json")
    m2 = SplitManifest.from_json(p)
    assert m2.folds[0].val == m.folds[0].val and len(m2.loco) == 3


def test_leakage_check_raises():
    a = CaseRecord("a", "P1", "Centre 1")
    b = CaseRecord("b", "P1", "Centre 3")
    with pytest.raises(ValueError):
        assert_no_leakage({"development": [a], "external": [b]})


def test_loco_folds():
    folds = leave_one_centre_out(_records())
    assert len(folds) == 3
    assert sum(len(f.val) for f in folds) == 60


# ------------------------------------------------------------------ preprocessing and io
def test_resample_crop_uncrop():
    vol = Volume(np.random.default_rng(0).random((20, 30, 40)).astype(np.float32), (0.5, 0.4, 0.3))
    iso = resample_isotropic(vol, 0.3)
    assert iso.spacing == (0.3, 0.3, 0.3)
    assert np.allclose(iso.extent_mm, vol.extent_mm, atol=0.31)
    crop, off = crop_cube(iso, iso.extent_mm / 2, size_mm=6.0)
    assert crop.shape == (20, 20, 20)
    back = uncrop(crop.array, off, iso.shape)
    m = back != 0
    assert np.allclose(back[m], iso.array[m])


def test_preprocessor_fit_on_training_only(tmp_path):
    ph = [make_phantom(shape=(48, 48, 48), seed=s) for s in range(2)]
    pre = Preprocessor(spacing=0.3).fit([p.image for p in ph], [p.labels.array for p in ph])
    out = pre.transform(ph[0].image)
    fg = out.array[ph[0].labels.array > 0]
    assert abs(float(fg.mean())) < 0.5
    p = pre.save(tmp_path / "pre.json")
    assert Preprocessor.load(p) == pre
    with pytest.raises(RuntimeError):
        Preprocessor().transform(ph[0].image)


def test_volume_and_json_io(tmp_path):
    vol = Volume(np.arange(24, dtype=np.float32).reshape(2, 3, 4), (1.0, 0.5, 0.25))
    p = save_volume(vol, tmp_path / "v.npz")
    v2 = load_volume(p)
    assert np.array_equal(v2.array, vol.array) and v2.spacing == vol.spacing
    q = write_json({"a": np.float32(1.5), "b": np.arange(3)}, tmp_path / "x.json")
    assert read_json(q) == {"a": 1.5, "b": [0, 1, 2]}


# ------------------------------------------------------------------ config
def test_config_layering_and_overrides():
    cfg = load_config("configs/experiment/development.yaml", ["ranking.top_k=5", "data.n_folds=10"])
    assert cfg.ranking.top_k == 5 and cfg.data.n_folds == 10
    assert cfg.data.external_centre == "Centre 3"
    assert parse_overrides(["a.b=1", "a.c=[1, 2]"]) == {"a": {"b": 1, "c": [1, 2]}}
    with pytest.raises(KeyError):
        load_config(None, ["ranking.not_a_key=1"])
