"""Unit tests for webplanner.geometry on synthetic bifurcation phantoms (spacing 0.3 mm, 96^3)."""
from __future__ import annotations

import time

import numpy as np
import pytest
from scipy import ndimage as ndi

from webplanner.data.synthetic import PhantomSpec, make_phantom, random_spec
from webplanner.geometry import analyze, direction_near, fit_neck_frame, measure, terminal_segment, vessel_origins
from webplanner.geometry.morphometry import digital_solidity, ostium_offset
from webplanner.geometry.neck import neck_mask
from webplanner.geometry.planning_factors import (Thresholds, angle_deg, assess, daughter_angle_difference,
                                                  entry_angle, is_simple_case, takeoff_angles)
from webplanner.types import BRANCH, NECK, SAC, NeckFrame, VesselOrigin, Volume, unit

SPECS = {
    "default": PhantomSpec(),
    "angled": PhantomSpec(width_1=7.0, width_2=6.0, height=6.5, neck_width=4.6, entry_angle_deg=70.0,
                          takeoff_deg=(95.0, 135.0), azimuth_deg=30.0),
    "bleb": PhantomSpec(bleb_radius=1.0),
    **{f"random{s}": random_spec(np.random.default_rng(s)) for s in (1, 6, 9, 11, 13, 20, 24, 25, 29)},
}
# Seeds are restricted to daughters with take-off angles of at least 85 deg. The phantom keeps branch voxels only
# below t = 0.6 mm (it removes them where they would enter the sac), so a daughter rising towards the sac is cut
# progressively: its top is sliced off at 80 deg (take-off error about 10 deg) and it is reduced to a sliver or removed
# below 70 deg (random_spec draws take-off angles down to 25 deg). The true angle is then not observable in the labels.
# Seeds 1, 6, 9, 24 and 25 have a daughter leaving backwards (take-off 145-165 deg); 9 and 20 have a bleb.
OBSERVABLE_TAKEOFF_DEG = 85.0

TOL = {"normal_deg": 8.0, "center_mm": 0.6, "mean_width": 0.6, "height": 0.7, "entry_deg": 12.0, "diff_deg": 15.0}
# neck_width is the largest diameter of the voxelised landing zone (physical extent, so up to one voxel wider
# than the analytic geometric-mean neck diameter).
NECK_TOL = 0.8


@pytest.fixture(scope="module")
def phantoms():
    assert all(min(v.takeoff_deg) >= OBSERVABLE_TAKEOFF_DEG for v in SPECS.values())
    return {k: make_phantom(v) for k, v in SPECS.items()}


@pytest.fixture(scope="module")
def results(phantoms):
    return {k: analyze(p.labels) for k, p in phantoms.items()}


def rotate_about_x(lab: np.ndarray, deg: float):
    """Rotate a label map in the (z, y) plane about the volume centre (nearest neighbour). Returns (labels, R)."""
    th = np.deg2rad(deg)
    R = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0], [0.0, 0.0, 1.0]])
    c = (np.asarray(lab.shape) - 1) / 2
    return ndi.affine_transform(lab, R.T, offset=c - R.T @ c, order=0, mode="constant", cval=0), R


# ---------------------------------------------------------------------------------------------------- neck frame
@pytest.mark.parametrize("name", list(SPECS))
def test_neck_frame_accuracy(name, phantoms, results):
    tr, fr = phantoms[name].truth, results[name].frame
    assert angle_deg(fr.normal, tr["normal"]) < TOL["normal_deg"]
    assert np.linalg.norm(fr.center - tr["neck_center"]) < TOL["center_mm"]
    basis = np.stack([fr.normal, fr.e1, fr.e2])
    assert np.allclose(basis @ basis.T, np.eye(3), atol=1e-9)
    assert np.allclose(np.cross(fr.normal, fr.e1), fr.e2)
    assert fr.eigenvalues[0] <= fr.eigenvalues[1] <= fr.eigenvalues[2]
    # the ostium plane lies below the landing-zone centroid, by about half the layer thickness
    assert -0.8 < ostium_offset(phantoms[name].labels, fr) < 0.0


def test_neck_frame_fallback_without_landing_zone(phantoms):
    ph = phantoms["default"]
    lab = ph.labels.array.copy()
    lab[lab == NECK] = SAC
    vol = Volume(lab, ph.labels.spacing)
    assert neck_mask(vol)[1] == "sac_artery_interface"
    fr = fit_neck_frame(vol)
    assert angle_deg(fr.normal, ph.truth["normal"]) < 10.0
    assert np.linalg.norm(fr.center - ph.truth["neck_center"]) < 1.0
    res = analyze(vol)
    assert abs(res.morphometry.mean_width - ph.truth["mean_width"]) < 0.8


def test_no_neck_raises():
    lab = np.zeros((20, 20, 20), np.uint8)
    lab[5:10, 5:10, 5:10] = SAC
    with pytest.raises(ValueError):
        fit_neck_frame(Volume(lab, (0.3, 0.3, 0.3)))


# ---------------------------------------------------------------------------------------------------- morphometry
@pytest.mark.parametrize("name", list(SPECS))
def test_morphometry_accuracy(name, phantoms, results):
    ph, m = phantoms[name], results[name].morphometry
    tr, s = ph.truth, ph.spec.spacing
    assert abs(m.mean_width - tr["mean_width"]) < TOL["mean_width"]
    assert abs(m.max_width - tr["max_width"]) < TOL["mean_width"]
    assert m.max_width >= max(m.width_1, m.width_2) - 1e-9
    assert abs(m.neck_width - tr["neck_width"]) < NECK_TOL
    assert m.width_neck_ratio == pytest.approx(m.mean_width / m.neck_width)
    assert m.volume_mm3 == pytest.approx(tr["sac_volume_mm3"])
    assert m.smallest_height <= m.height + 1e-9
    if tr["irregular"]:
        # the truth height describes the ellipsoid; a bleb can only add height
        assert m.height > tr["height"] - TOL["height"]
    else:
        assert abs(m.height - tr["height"]) < TOL["height"]
        assert abs(m.smallest_height - tr["height"]) < TOL["height"]
        # h_sac runs from the frame centre (the landing-zone centroid), about half a layer above the neck plane
        t_center = (results[name].frame.center - tr["neck_center"]) @ tr["normal"]
        assert abs(m.sac_axis_extent - (tr["height"] - t_center)) < TOL["height"]
        assert abs(m.sac_axis_extent - ostium_offset(ph.labels, results[name].frame) - tr["height"]) < TOL["height"]
    assert not m.branch_incorporated


def test_solidity_regular_vs_bleb(results):
    regular, bleb = results["default"].morphometry, results["bleb"].morphometry
    assert regular.solidity > 0.9
    assert regular.solidity == pytest.approx(1.0)     # a digitised convex sac is digitally convex
    assert bleb.solidity < regular.solidity
    assert not regular.irregular


def test_irregular_threshold_is_configurable(phantoms, results):
    ph = phantoms["bleb"]
    fr, orr = results["bleb"].frame, results["bleb"].origins
    sol = results["bleb"].morphometry.solidity
    assert measure(ph.labels, fr, orr, irregular_threshold=sol + 1e-3).irregular
    assert not measure(ph.labels, fr, orr, irregular_threshold=sol - 1e-3).irregular


def test_digital_solidity_degenerate():
    m = np.zeros((10, 10, 10), bool)
    m[5, 2:8, 2:8] = True                            # a flat plate
    assert digital_solidity(m, (1.0, 1.0, 1.0)) == 1.0
    m = np.zeros((10, 10, 10), bool)
    m[2:8, 2:8, 2:8] = True
    m[4:6, 4:6, 2:8] = False                         # a tunnel: not convex
    assert digital_solidity(m, (1.0, 1.0, 1.0)) < 1.0


def test_branch_incorporation(phantoms, results):
    ph = phantoms["default"]
    tr, s = ph.truth, ph.spec.spacing
    lab = ph.labels.array.copy()
    el = tr["ellipsoid"]
    P = np.stack(np.meshgrid(*[np.arange(n) * s for n in lab.shape], indexing="ij"), axis=-1)
    wall = tr["neck_center"] + el["t0"] * tr["normal"] + el["a2"] * tr["e2"]     # equator of the sac along e2
    d = P - wall
    along = d @ tr["e2"]
    radial = np.linalg.norm(d - along[..., None] * tr["e2"], axis=-1)
    tube = (radial <= 0.9) & (along >= -1.0) & (along <= 6.0) & (lab == 0)
    lab[tube] = BRANCH
    vol = Volume(lab, ph.labels.spacing)
    fr = results["default"].frame
    assert measure(vol, fr, vessel_origins(vol, fr)).branch_incorporated


# ---------------------------------------------------------------------------------------------- vessels and angles
@pytest.mark.parametrize("name", list(SPECS))
def test_planning_angles(name, phantoms, results):
    tr, res = phantoms[name].truth, results[name]
    names = [o.name for o in res.origins]
    assert names == ["parent", "branch_1", "branch_2"]
    assert abs(res.factors.entry_angle_deg - tr["entry_angle_deg"]) < TOL["entry_deg"]
    assert abs(res.factors.daughter_angle_diff_deg - tr["daughter_angle_diff_deg"]) < TOL["diff_deg"]
    # each daughter direction matches one true daughter direction, with the correct take-off angle
    dirs = tr["daughter_dirs"]
    for o in res.origins[1:]:
        k = int(np.argmax([o.direction @ dv for dv in dirs]))
        # random24: the daughter leaving at 160 deg runs back along the parent artery, which overwrites part of it
        assert abs(angle_deg(o.direction, res.frame.normal) - tr["takeoff_deg"][k]) < TOL["entry_deg"]
    for o in res.origins:
        assert np.linalg.norm(o.direction) == pytest.approx(1.0)
        assert 0.3 < o.radius_mm < 2.5


def test_vessel_radius_and_order(phantoms, results):
    spec, res = phantoms["default"].spec, results["default"]
    parent, b1, b2 = res.origins
    assert abs(parent.radius_mm - spec.parent_radius) < 0.3
    assert abs(b1.radius_mm - spec.branch_radii[0]) < 0.3
    assert abs(b2.radius_mm - spec.branch_radii[1]) < 0.3


def test_vessel_origins_pca_method(phantoms, results):
    ph, fr = phantoms["default"], results["default"].frame
    orr = vessel_origins(ph.labels, fr, method="pca")
    assert [o.name for o in orr] == ["parent", "branch_1", "branch_2"]
    for o in orr:
        assert np.linalg.norm(o.direction) == pytest.approx(1.0)
        # points away from the neck
        assert o.direction @ (o.point - fr.center) > -0.5
    with pytest.raises(ValueError):
        vessel_origins(ph.labels, fr, method="spline")


def test_significant_angulation_flag(results):
    f = results["angled"].factors
    assert f.significant_angulation and not f.simple_case
    assert any("angulation" in x for x in f.simple_case_failures)
    assert not results["default"].factors.significant_angulation


def test_rotated_phantom(phantoms):
    """The method does not rely on the neck being aligned with the voxel grid."""
    ph = phantoms["default"]
    tr, s = ph.truth, ph.spec.spacing
    lab, R = rotate_about_x(ph.labels.array, 30.0)
    c = (np.asarray(lab.shape) - 1) / 2 * s
    res = analyze(Volume(lab, ph.labels.spacing))
    assert angle_deg(res.frame.normal, R @ tr["normal"]) < TOL["normal_deg"]
    assert np.linalg.norm(res.frame.center - (R @ (tr["neck_center"] - c) + c)) < TOL["center_mm"]
    assert abs(res.morphometry.mean_width - tr["mean_width"]) < TOL["mean_width"]
    assert abs(res.morphometry.height - tr["height"]) < TOL["height"]
    assert abs(res.factors.entry_angle_deg - tr["entry_angle_deg"]) < TOL["entry_deg"]
    assert abs(res.factors.daughter_angle_diff_deg - tr["daughter_angle_diff_deg"]) < TOL["diff_deg"]


def test_axis_permutation_invariance(phantoms, results):
    ph, ref = phantoms["random13"], results["random13"]
    perm = Volume(np.ascontiguousarray(ph.labels.array.transpose(2, 0, 1)), ph.labels.spacing)
    res = analyze(perm)
    assert angle_deg(res.frame.normal, ref.frame.normal[[2, 0, 1]]) < 1e-6
    for key in ("mean_width", "max_width", "height", "neck_width", "volume_mm3", "solidity", "sac_axis_extent"):
        assert getattr(res.morphometry, key) == pytest.approx(getattr(ref.morphometry, key), abs=1e-6)
    assert res.factors.entry_angle_deg == pytest.approx(ref.factors.entry_angle_deg, abs=3.0)


# ---------------------------------------------------------------------------------------------------- centreline
def _tube(direction, radius=1.0, shape=(64, 64, 64), s=0.3):
    P = np.stack(np.meshgrid(*[np.arange(n) * s for n in shape], indexing="ij"), axis=-1)
    a = np.asarray(shape) * s * 0.25
    d = P - a
    along = d @ direction
    radial = np.linalg.norm(d - along[..., None] * direction, axis=-1)
    return (radial <= radius) & (along >= 0) & (along <= 14.0), a


def test_centerline_direction_of_straight_tube():
    d = unit([0.3, 0.5, 0.8])
    mask, a = _tube(d)
    sp = (0.3, 0.3, 0.3)
    seg = terminal_segment(a, mask, sp)
    assert seg.method == "skeleton"
    assert angle_deg(seg.direction, d) < 3.0
    assert abs(seg.radius_mm - 1.0) < 0.15
    e1 = unit(np.cross(d, [1.0, 0.0, 0.0]))
    frame = NeckFrame(center=a - d, normal=-d, e1=e1, e2=np.cross(-d, e1))
    assert angle_deg(direction_near(a, Volume(mask, sp), frame), d) < 3.0
    assert angle_deg(direction_near(a, mask, frame, towards_neck=True, spacing=sp), -d) < 3.0
    pca = terminal_segment(a, mask, sp, method="pca")
    assert pca.method == "pca" and angle_deg(pca.direction, d) < 5.0
    assert abs(pca.radius_mm - 1.0) < 0.3


def test_centerline_noisy_tube():
    d = unit([0.6, -0.2, 0.7])
    mask, a = _tube(d, radius=1.2)
    rng = np.random.default_rng(0)
    noisy = mask ^ (ndi.binary_dilation(mask) & ~mask & (rng.random(mask.shape) < 0.3))   # bumpy wall
    noisy &= ~(rng.random(mask.shape) < 0.02)                                             # pin holes
    seg = terminal_segment(a, noisy, (0.3, 0.3, 0.3))
    assert angle_deg(seg.direction, d) < 6.0


# ---------------------------------------------------------------------------------------------------- planning factors
def test_angle_functions():
    n = np.array([1.0, 0.0, 0.0])
    assert entry_angle(n, n) == pytest.approx(0.0)
    assert entry_angle(np.array([1.0, 1.0, 0.0]), n) == pytest.approx(45.0)
    assert takeoff_angles([np.array([0.0, 1.0, 0.0]), np.array([-1.0, 1.0, 0.0])], n) == pytest.approx([90.0, 135.0])
    assert np.isnan(entry_angle(np.zeros(3), n))
    diff, note = daughter_angle_difference([100.0, 130.0])
    assert diff == pytest.approx(30.0) and note is None
    diff, note = daughter_angle_difference([100.0])
    assert diff == 0.0 and "not assessable" in note


def test_is_simple_case():
    ok, failed = is_simple_case(6.0, 1.2, 10.0, 20.0, ruptured=False, tortuosity="minimal")
    assert ok and failed == []
    notes = []
    ok, failed = is_simple_case(6.0, 1.2, 10.0, 20.0, notes=notes)
    assert ok and len(notes) == 2                     # unknown rupture and tortuosity: noted, not failed
    ok, failed = is_simple_case(11.0, 1.8, 50.0, 70.0, ruptured=True, tortuosity="severe")
    assert not ok and len(failed) == 6
    ok, failed = is_simple_case(6.0, 1.2, 10.0, 20.0, ruptured=False, tortuosity="minimal",
                                thresholds=Thresholds(width_range=(3.0, 5.0)))
    assert not ok and failed[0].startswith("width")


def test_assess_with_one_daughter_and_missing_direction(phantoms, results):
    ph, res = phantoms["default"], results["default"]
    parent, b1, _ = res.origins
    broken = VesselOrigin(name="parent", point=parent.point, direction=np.full(3, np.nan), radius_mm=1.0)
    f = assess(ph.labels, res.frame, [broken, b1], res.morphometry, ruptured=False, tortuosity="minimal")
    assert f.entry_angle_deg == pytest.approx(res.factors.entry_angle_deg, abs=1.0)
    assert f.daughter_angle_diff_deg == 0.0 and not f.asymmetric_daughters
    assert any("daughter" in n for n in f.notes)
    f = assess(ph.labels, res.frame, [b1], res.morphometry)
    assert np.isnan(f.entry_angle_deg) and not f.significant_angulation
    assert any("parent" in n for n in f.notes)


def test_analyze_speed_and_inputs(phantoms):
    ph = phantoms["default"]
    t0 = time.perf_counter()
    res = analyze(ph.labels, ruptured=True, tortuosity="minimal")
    assert time.perf_counter() - t0 < 5.0
    assert res.factors.ruptured is True and "ruptured" in res.factors.simple_case_failures
    assert res.morphometry.to_dict()["irregular"] is False
