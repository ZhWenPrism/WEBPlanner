"""Graphical abstract computed by the planner on a synthetic phantom (no patient data).

Panels: anatomy labels, neck frame and morphometry, virtual deployment of the top-ranked size, dome clearance over
the catalogue, and the ranked sizes. The ranking model is trained on a synthetic cohort, so its probabilities only
illustrate the output format.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from webplanner.data.synthetic import PhantomSpec, make_phantom, operator_choice, synthetic_cases  # noqa: E402
from webplanner.device.catalogue import Catalogue  # noqa: E402
from webplanner.device.deployment import VirtualDeployer, deploy_catalogue  # noqa: E402
from webplanner.pipeline import WEBPlanner, analyse_cases, build_candidates, train_ranker  # noqa: E402
from webplanner.ranking import FeatureBuilder  # noqa: E402
from webplanner.types import CaseRecord  # noqa: E402

INK, MUTED = "#1f2937", "#6b7280"
LAB_CMAP = ListedColormap(["#00000000", "#e11d48", "#2563eb", "#0891b2", "#f59e0b"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "mathtext.fontset": "dejavusans"})


def _train_synthetic_ranker(cat: Catalogue, deployer: VirtualDeployer, n: int = 48, seed: int = 3):
    rng = np.random.default_rng(seed)
    cases, ph = [], {}
    for rec, spec in synthetic_cases(n, seed=seed):
        spec.shape = (80, 80, 80)
        p = make_phantom(spec)
        t = p.truth
        rec.implanted = operator_choice(t["mean_width"], t["height"], t["neck_width"], t["irregular"], t["entry_angle_deg"], cat, rng)
        ph[rec.case_id] = p
        cases.append(rec)
    arts, _ = analyse_cases(cases, lambda r: ph[r.case_id].labels, cat, deployer)
    builder = FeatureBuilder(cat, image_score=False)
    tr = train_ranker(build_candidates(arts, builder), grid=[0.3, 3.0, 30.0], n_splits=3)
    return WEBPlanner(tr.model, tr.review, cat, deployer, builder)


def _slice_along_axis(vol: np.ndarray, frame, spacing: float, half_mm: float = 9.0, step: float = 0.1, axis_vec=None):
    """Resample the plane spanned by the neck normal and e1 through the neck centre."""
    from scipy import ndimage as ndi

    e = frame.e1 if axis_vec is None else axis_vec
    ts = np.arange(-3.0, 11.0, step)
    us = np.arange(-half_mm, half_mm, step)
    T, U = np.meshgrid(ts, us, indexing="ij")
    P = frame.center + T[..., None] * frame.normal + U[..., None] * e
    return ndi.map_coordinates(vol.astype(np.float32), (P / spacing).reshape(-1, 3).T, order=0).reshape(T.shape), ts, us


def graphical_abstract(out: Path) -> Path:
    cat = Catalogue.default()
    deployer = VirtualDeployer(k=0.45)
    planner = _train_synthetic_ranker(cat, deployer)
    spec = PhantomSpec(shape=(96, 96, 96), width_1=7.0, width_2=6.2, height=6.2, neck_width=4.4, entry_angle_deg=35.0,
                       takeoff_deg=(95.0, 120.0), bleb_radius=0.9, bleb_polar_deg=40.0, seed=5)
    ph = make_phantom(spec)
    rec = CaseRecord("phantom", "phantom", "synthetic", location="MCA", ruptured=False, access_tortuosity="minimal")
    rep = planner.plan(rec, image=ph.image, labels=ph.labels)
    a = rep.anatomy
    deps = deploy_catalogue(ph.labels, a.frame, cat, deployer)
    top = rep.top[0]
    dep_top = deployer.deploy(ph.labels, a.frame, top.size, return_mask=True)
    s = ph.spec.spacing

    fig = plt.figure(figsize=(16, 5.0), dpi=200)
    fig.text(0.012, 0.965, "WEBPlanner on a synthetic bifurcation aneurysm", fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.905, "every number below is computed by the code in this repository; the phantom and the synthetic ranking model "
             "contain no patient data", fontsize=8.5, color=MUTED, va="top")
    W = [0.02, 0.215, 0.41, 0.625, 0.815]
    axw, B, H, TY = 0.165, 0.15, 0.62, 0.815

    def title(i, t):
        fig.text(W[i], TY, t, fontsize=10.5, fontweight="bold", color=INK, va="bottom")

    # A: labels (maximum projection along y)
    ax = fig.add_axes([W[0], B, axw, H])
    lab = ph.labels.array
    proj = np.zeros((lab.shape[0], lab.shape[2]), np.uint8)
    for c in (2, 3, 1, 4):
        proj[(lab == c).any(axis=1)] = c
    cz, cx = int(round(a.frame.center[0] / s)), int(round(a.frame.center[2] / s))
    h = int(round(10.0 / s))
    crop = proj[max(cz - h, 0):cz + h, max(cx - h, 0):cx + h]
    ax.imshow(crop, cmap=LAB_CMAP, vmin=0, vmax=4, interpolation="nearest", origin="lower")
    title(0, "A  Four-class anatomy")
    ax.axis("off")
    leg = [("sac", "#e11d48"), ("parent artery", "#2563eb"), ("branches", "#0891b2"), ("landing zone", "#f59e0b")]
    for i, (t, c) in enumerate(leg):
        fig.text(W[0] + (i % 2) * 0.085, 0.085 - (i // 2) * 0.04, "■ " + t, color=c, fontsize=8)

    # B: neck frame and morphometry (section through the axis)
    ax = fig.add_axes([W[1], B, axw, H])
    sec, ts, us = _slice_along_axis(lab, a.frame, s)
    ax.imshow(sec, cmap=LAB_CMAP, vmin=0, vmax=4, origin="lower", extent=[us[0], us[-1], ts[0], ts[-1]], interpolation="nearest")
    m = a.morphometry
    ax.annotate("", xy=(0, m.height), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    ax.text(0.3, m.height * 0.55, r"$\mathbf{n}$", fontsize=11, color=INK)
    ax.plot([-m.neck_width / 2, m.neck_width / 2], [0, 0], color=INK, lw=1.0, ls="--")
    title(1, "B  Neck plane and morphometry")
    txt = (f"mean width {m.mean_width:.1f} mm\nheight {m.height:.1f} mm\nneck {m.neck_width:.1f} mm\n"
           f"entry angle {a.factors.entry_angle_deg:.0f}°\nsolidity {m.solidity:.2f}")
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, va="top", fontsize=7.8, color=INK,
            bbox=dict(fc="white", ec="#d1d5db", alpha=0.9, pad=3))
    ax.set_xlabel("mm", fontsize=7)
    ax.tick_params(labelsize=6.5)

    # C: virtual deployment of the top-1 size
    ax = fig.add_axes([W[2], B, axw, H])
    dsec, _, _ = _slice_along_axis(dep_top.mask.astype(np.float32), a.frame, s)
    ax.imshow(np.isin(sec, (1, 4)), cmap=ListedColormap(["white", "#fecdd3"]), origin="lower", extent=[us[0], us[-1], ts[0], ts[-1]])
    ax.contour(us, ts, np.isin(sec, (1, 4)).astype(float), levels=[0.5], colors="#e11d48", linewidths=1.0)
    ax.contourf(us, ts, dsec, levels=[0.5, 1.5], colors=["#1f2937"], alpha=0.55)
    title(2, f"C  Virtual deployment, WEB {top.size.label}")
    d = dep_top
    ax.text(0.02, 0.98, f"c = {d.compression:.2f}\nH′ = {d.deployed_height:.1f} mm\ng = {d.dome_clearance:.1f} mm\n"
            f"apposition {d.apposition:.2f}\nneck coverage {d.neck_coverage:.2f}", transform=ax.transAxes, va="top", fontsize=7.8,
            color=INK, bbox=dict(fc="white", ec="#d1d5db", alpha=0.9, pad=3))
    ax.set_xlabel("mm", fontsize=7)
    ax.tick_params(labelsize=6.5)

    # D: dome clearance over the catalogue
    ax = fig.add_axes([W[3] + 0.03, B, axw - 0.03, H])
    widths = list(cat.widths)
    heights = sorted({sz.height for sz in cat.sizes})
    G = np.full((len(heights), len(widths)), np.nan)
    for sz, dd in zip(cat.sizes, deps):
        G[heights.index(sz.height), widths.index(sz.width)] = dd.dome_clearance
    ax.imshow(G, origin="lower", cmap="RdYlGn", vmin=-1.5, vmax=1.5, aspect="auto")
    for r in rep.top:
        ax.add_patch(Rectangle((widths.index(r.size.width) - 0.5, heights.index(r.size.height) - 0.5), 1, 1, fill=False, ec=INK, lw=1.5))
    c = rep.chart_size
    ax.plot(widths.index(c.width), heights.index(c.height), marker="x", color=INK, ms=7, mew=1.6)
    ax.set_xticks(range(len(widths)), [f"{w:g}" for w in widths], fontsize=6.5)
    ax.set_yticks(range(len(heights)), [f"{h:g}" for h in heights], fontsize=6.5)
    ax.set_xlabel("device width (mm)", fontsize=7.5)
    ax.set_ylabel("device height (mm)", fontsize=7.5)
    title(3, "D  Dome clearance g (mm)")
    fig.text(W[3] + 0.03, 0.035, "boxes: top-3 sizes    ×: sizing chart", fontsize=7.5, color=MUTED)

    # E: ranking
    ax = fig.add_axes([W[4] + 0.055, B, axw - 0.05, H])
    ps = [r.probability for r in rep.top]
    ax.barh(range(len(ps))[::-1], ps, color=["#2563eb", "#93b4f5", "#c7d7fa"][: len(ps)], height=0.6)
    for i, r in enumerate(rep.top):
        ax.text(-0.02, len(ps) - 1 - i, f"WEB {r.size.label}", ha="right", va="center", fontsize=8.5)
        ax.text(r.probability + 0.02, len(ps) - 1 - i, f"{r.probability:.2f}", va="center", fontsize=8, color=MUTED)
    ax.set_xlim(0, 1)
    ax.axvline(rep.tau, color="#9ca3af", ls=":", lw=1)
    ax.text(rep.tau, len(ps) - 0.45, r" $\tau$", fontsize=8, color=MUTED)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(labelsize=6.5)
    ax.set_xlabel("P(size | case)", fontsize=7.5)
    title(4, "E  Ranked sizes")
    fig.text(W[4], 0.035, f"review flag: {'yes' if rep.review_flag else 'no'} (synthetic ranking model)", fontsize=7.5, color=MUTED)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(graphical_abstract(ROOT / "assets" / "graphical-abstract.png"))
