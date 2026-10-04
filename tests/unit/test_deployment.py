"""Virtual deployment (eq. 1), elongation fit and deployment-stage rendering on synthetic phantoms."""
from __future__ import annotations

import time

import numpy as np
import pytest
from scipy import ndimage as ndi

from webplanner.data.synthetic import make_phantom
from webplanner.device import Catalogue, VirtualDeployer, deploy_catalogue, fit_elongation, predict_height
from webplanner.device.deployment import (DeploymentConfig, axis_extent, device_dice, device_span, nominal_profile,
                                          sac_radius)
from webplanner.device.fluoro import STAGES, marker_positions, projection_view, render_stages, stage_geometry
from webplanner.types import NECK, SAC, DeviceSize, NeckFrame, Volume


def truth_frame(truth) -> NeckFrame:
    return NeckFrame(center=truth["neck_center"], normal=truth["normal"], e1=truth["e1"], e2=truth["e2"])


@pytest.fixture(scope="module")
def phantom():
    return make_phantom(shape=(80, 80, 80))


@pytest.fixture(scope="module")
def deployer():
    return VirtualDeployer()


@pytest.fixture(scope="module")
def field(phantom, deployer):
    return deployer.prepare(phantom.labels, truth_frame(phantom.truth))


# --------------------------------------------------------------------------------------------------------------------
# sac geometry
# --------------------------------------------------------------------------------------------------------------------
def test_axis_extent_matches_phantom(phantom, field):
    assert abs(axis_extent(field) - phantom.truth["height"]) < 0.5
    assert abs(field.t_entry) < 0.3


def test_sac_radius_matches_ellipsoid(phantom, field):
    el = phantom.truth["ellipsoid"]
    r = sac_radius(field, [el["t0"]], [0.0, np.pi / 2, np.pi, 3 * np.pi / 2])[0]
    assert np.allclose(r, [el["a1"], el["a2"], el["a1"], el["a2"]], atol=0.3)
    assert np.all(sac_radius(field, [-1.0], field.angles) == 0.0)        # axis below the neck plane is outside the sac


def test_nominal_profile_shape():
    z = np.linspace(0, 4.0, 41)
    r = nominal_profile(8.0, 4.0, z)
    assert r[0] == pytest.approx(0.0) and r[-1] == pytest.approx(0.0, abs=1e-12)
    assert r.max() == pytest.approx(4.0) and np.argmax(r) == 20
    assert np.all(nominal_profile(8.0, 4.0, z, exponent=2.0) <= r + 1e-12)   # SLS is rounder than the SL barrel


# --------------------------------------------------------------------------------------------------------------------
# equation 1
# --------------------------------------------------------------------------------------------------------------------
def test_compression_increases_with_width(field, deployer):
    c = [deployer.deploy(field, None, DeviceSize(w, 4.0)).compression for w in (5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0)]
    assert np.all(np.diff(c) >= -1e-9)
    assert c[-1] > c[1] + 0.3


def test_narrow_device_is_not_compressed(field, deployer):
    r = deployer.deploy(field, None, DeviceSize(4.0, 2.0))
    assert r.compression < 0.05
    assert r.deployed_width == pytest.approx(4.0, abs=0.05)


def test_wide_device_conforms_to_sac(phantom, field, deployer):
    for size in (DeviceSize(9.0, 5.0), DeviceSize(11.0, 6.0)):
        r = deployer.deploy(field, None, size)
        assert r.compression > 0.3
        assert abs(r.deployed_width - phantom.truth["mean_width"]) < 0.6
        assert r.apposition > 0.8


def test_height_rule_uncapped_and_capped(field, deployer):
    k = deployer.config.k
    r = deployer.deploy(field, None, DeviceSize(7.0, 3.0))
    assert r.free_height < r.h_sac
    assert r.free_height == pytest.approx(3.0 + k * 7.0 * r.compression, abs=1e-9)
    assert r.deployed_height == pytest.approx(r.free_height, abs=1e-9)
    assert r.dome_clearance == r.h_sac - r.deployed_height

    r = deployer.deploy(field, None, DeviceSize(10.0, 6.0))
    assert r.free_height == pytest.approx(6.0 + k * 10.0 * r.compression, abs=1e-9)
    assert r.free_height > r.h_sac >= 0.8 * 6.0
    assert r.deployed_height == pytest.approx(r.h_sac, abs=1e-12)
    assert r.dome_clearance == r.h_sac - r.deployed_height


def test_floor_and_protrusion(field, deployer):
    cfg = deployer.config
    t_ref = min(0.0, field.t_entry)
    normal = deployer.deploy(field, None, DeviceSize(7.0, 3.0))
    assert normal.protrusion == pytest.approx(max(0.0, t_ref - cfg.base_offset_mm))
    tall = deployer.deploy(field, None, DeviceSize(11.0, 9.0))      # h_sac < 0.8 H_d: floor applies
    assert tall.deployed_height == pytest.approx(cfg.min_height_fraction * 9.0)
    assert tall.dome_clearance == tall.h_sac - tall.deployed_height < 0
    t_base, t_top = device_span(tall, cfg)
    assert t_top == pytest.approx(tall.h_sac)
    assert tall.protrusion == pytest.approx(t_ref - t_base) and tall.protrusion > normal.protrusion


def test_neck_coverage(field, deployer):
    wide = deployer.deploy(field, None, DeviceSize(8.0, 4.0))
    assert wide.neck_coverage > 0.98 and wide.neck_margin > 0
    narrow = deployer.deploy(field, None, DeviceSize(4.0, 3.0))
    assert narrow.neck_coverage < wide.neck_coverage and narrow.neck_margin < 0


def test_frame_permutation_invariance(phantom, deployer):
    """Same anatomy with the array axes permuted gives the same deployment."""
    t = phantom.truth
    perm = lambda v: np.asarray(v, float)[::-1]
    lab2 = Volume(phantom.labels.array.transpose(2, 1, 0).copy(), phantom.labels.spacing[::-1])
    fr2 = NeckFrame(center=perm(t["neck_center"]), normal=perm(t["normal"]), e1=perm(t["e1"]), e2=perm(t["e2"]))
    a = deployer.deploy(phantom.labels, truth_frame(t), DeviceSize(8.0, 4.0))
    b = deployer.deploy(lab2, fr2, DeviceSize(8.0, 4.0))
    for name in ("compression", "deployed_width", "deployed_height", "apposition", "neck_coverage", "neck_margin", "h_sac"):
        assert getattr(a, name) == pytest.approx(getattr(b, name), abs=1e-5)


def test_deploy_catalogue_order_and_speed(phantom):
    cat = Catalogue.default()
    t0 = time.perf_counter()
    res = deploy_catalogue(phantom.labels, truth_frame(phantom.truth), cat)
    elapsed = time.perf_counter() - t0
    assert len(res) == 29 == len(cat)
    assert [r.size for r in res] == cat.sizes
    assert elapsed < 10.0
    for r in res:
        assert all(np.isfinite(v) for v in r.features().values())


def test_mask_inside_sac(phantom, field, deployer):
    r = deployer.deploy(field, None, DeviceSize(9.0, 5.0), return_mask=True)
    m = r.mask
    assert m.shape == phantom.labels.shape and m.dtype == bool and m.sum() > 500
    sac = np.isin(phantom.labels.array, (SAC, NECK))
    near = ndi.binary_dilation(sac, iterations=2)
    assert (m & near).sum() / m.sum() > 0.97
    assert (m & sac).sum() / m.sum() > 0.85
    other = deployer.deploy(field, None, DeviceSize(5.0, 2.0), return_mask=True).mask
    assert device_dice(m, m) == 1.0
    assert 0.0 < device_dice(m, other) < 1.0


def test_landing_zone_fallback(phantom, deployer):
    lab = phantom.labels.array.copy()
    lab[lab == NECK] = SAC
    with pytest.warns(UserWarning):
        fld = deployer.prepare(Volume(lab, phantom.labels.spacing), truth_frame(phantom.truth))
    assert fld.lz_source == "lowest_sac_layer"
    r = deployer.deploy(fld, None, DeviceSize(8.0, 4.0))
    assert np.isfinite(r.neck_margin) and r.neck_coverage > 0.9


def test_config_overrides():
    d = VirtualDeployer(k=0.6, n_angles=48)
    assert d.config.k == 0.6 and d.config.n_angles == 48 and d.config.n_levels == DeploymentConfig().n_levels
    with pytest.raises(TypeError):
        VirtualDeployer(not_a_parameter=1.0)


# --------------------------------------------------------------------------------------------------------------------
# elongation fit
# --------------------------------------------------------------------------------------------------------------------
def test_fit_elongation_recovers_k():
    rng = np.random.default_rng(3)
    n = 80
    cat = Catalogue.default()
    sizes = [cat.sizes[i] for i in rng.integers(0, len(cat), n)]
    W = np.array([s.width for s in sizes])
    H = np.array([s.height for s in sizes])
    c = rng.uniform(0.05, 0.45, n)
    Hobs = predict_height(W, H, c, k=0.45) + rng.normal(0.0, 0.25, n)
    fit = fit_elongation(W, H, c, Hobs, n_boot=2000, seed=0)
    assert abs(fit.k - 0.45) < 0.05
    assert fit.ci_low < 0.45 < fit.ci_high and fit.ci_low < fit.k < fit.ci_high
    assert fit.n == n and 0.15 < fit.rmse < 0.35
    groups = np.repeat(np.arange(n // 2), 2)                          # two devices per patient
    fit_g = fit_elongation(W, H, c, Hobs, n_boot=500, seed=1, groups=groups)
    assert fit_g.k == pytest.approx(fit.k) and fit_g.n_groups == n // 2
    with pytest.raises(ValueError):
        fit_elongation(W, H, np.zeros(n), Hobs)


# --------------------------------------------------------------------------------------------------------------------
# deployment-stage rehearsal
# --------------------------------------------------------------------------------------------------------------------
def test_render_stages_markers_visible(field, deployer):
    dep = deployer.deploy(field, None, DeviceSize(7.0, 4.0))
    imgs = render_stages(field, None, dep, deployer=deployer)
    assert tuple(imgs) == STAGES
    geoms = stage_geometry(deployer.profile(field, dep.size))
    view = projection_view(field.frame, geoms.values())
    fr = field.frame
    for name, img in imgs.items():
        assert img.ndim == 2 and img.shape[0] == img.shape[1] and np.all(np.isfinite(img))
        peak = np.array(np.unravel_index(np.argmax(img), img.shape))
        px = [view.to_pixel(fr.to_array(np.array([t, 0.0, 0.0])))[0] for t in geoms[name].markers.values()]
        assert min(np.linalg.norm(peak - p) for p in px) <= 2.0
        for p in px:                                                   # both markers stand out from the device
            r, c = np.round(p).astype(int)
            assert img[r, c] > 2.0 * np.percentile(img, 99)

    m = marker_positions(dep, fr, deployer.config)
    sep = (m["distal"] - m["proximal"]) @ fr.normal
    assert sep == pytest.approx(dep.deployed_height - 0.45, abs=1e-9)
