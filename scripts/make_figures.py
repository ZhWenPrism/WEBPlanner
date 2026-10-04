"""Figures for the README: the architecture schematic and a graphical abstract computed on a synthetic phantom.

    python scripts/make_figures.py            # writes assets/architecture.png and assets/graphical-abstract.png

No patient data are used: the graphical abstract runs the planner on a phantom from webplanner.data.synthetic.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
LEARN, LEARN_BG = "#2563eb", "#eaf1fe"        # learned models
COMP, COMP_BG = "#b45309", "#fdf3e6"          # computation (geometry, simulation)
EVAL, EVAL_BG = "#047857", "#e8f6f0"
CLASS_COL = {"sac": "#e11d48", "parent": "#2563eb", "branch": "#0891b2", "neck": "#f59e0b"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "mathtext.fontset": "dejavusans"})


def box(ax, x, y, w, h, fc, ec, lw=1.2, r=0.012, z=1):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z))


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.4, ms=11):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw, zorder=5))


def header(ax, x, y, w, title, sub, color):
    ax.text(x + 0.012, y, title, fontsize=11.5, fontweight="bold", color=color, va="top")
    ax.text(x + 0.012, y - 0.036, sub, fontsize=8.2, color=MUTED, va="top")


def architecture(out: Path) -> Path:
    fig = plt.figure(figsize=(16, 7.6), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.015, 0.975, "WEBPlanner", fontsize=17, fontweight="bold", color=INK, va="top")
    ax.text(0.13, 0.968, "3D rotational angiography  →  ranked Woven EndoBridge sizes with virtual deployment and a review flag",
            fontsize=10.5, color=MUTED, va="top")

    top, bot = 0.90, 0.30
    cols = [(0.015, 0.105), (0.135, 0.205), (0.355, 0.205), (0.575, 0.225), (0.815, 0.17)]

    # ---------------- input
    x, w = cols[0]
    box(ax, x, bot, w, top - bot, "#f8fafc", "#94a3b8")
    header(ax, x, top - 0.015, w, "Input", "pre-treatment", INK)
    vol = np.zeros((40, 40))
    yy, xx = np.mgrid[0:40, 0:40]
    vol += np.exp(-(((xx - 20) / 6.5) ** 2 + ((yy - 15) / 6.0) ** 2))
    vol += 0.8 * np.exp(-((xx - 20 - 0.15 * (yy - 26)) / 2.2) ** 2) * (yy > 22)
    vol += 0.7 * np.exp(-((yy - 23 + 0.35 * (xx - 20)) / 2.0) ** 2) * (np.abs(xx - 20) > 3)
    ia = fig.add_axes([x + 0.012, 0.56, w - 0.024, 0.2])
    ia.imshow(np.clip(vol, 0, 1), cmap="gray", origin="upper")
    ia.set_xticks([]), ia.set_yticks([])
    for s in ia.spines.values():
        s.set_color("#94a3b8")
    ax.text(x + w / 2, 0.535, "3D-RA volume", ha="center", fontsize=9, fontweight="bold", color=INK)
    ax.text(x + w / 2, 0.505, "isotropic 0.27–0.46 mm\n40 mm aneurysm-centred cube", ha="center", va="top", fontsize=7.6, color=MUTED)
    ax.text(x + w / 2, 0.40, "+ rupture status\n+ access tortuosity\n(from records)", ha="center", va="top", fontsize=7.6, color=MUTED)

    # ---------------- perception
    x, w = cols[1]
    box(ax, x, bot, w, top - bot, LEARN_BG, LEARN)
    header(ax, x, top - 0.015, w, "1  Anatomy", "learned: 3D segmentation", LEARN)
    # encoder-decoder schematic
    ex, ey = x + 0.02, 0.60
    hs = [0.17, 0.13, 0.09, 0.06]
    for i, h in enumerate(hs):
        ax.add_patch(Rectangle((ex + i * 0.019, ey + (0.17 - h) / 2), 0.012, h, fc=LEARN, ec="none", alpha=0.35 + 0.15 * i, zorder=3))
        ax.add_patch(Rectangle((ex + (7 - i) * 0.019, ey + (0.17 - h) / 2), 0.012, h, fc=LEARN, ec="none", alpha=0.35 + 0.15 * i, zorder=3))
        if i < 3:
            ax.plot([ex + i * 0.019 + 0.012, ex + (7 - i) * 0.019], [ey + 0.17 - (0.17 - h) / 2 + 0.006] * 2, color=LEARN, lw=0.8, ls=":", zorder=2)
    ax.text(x + w / 2, 0.585, "nnU-Net 3d_fullres (residual encoder)\n5-fold ensemble, mirroring TTA", ha="center", va="top", fontsize=7.8, color=INK)
    ax.text(x + w / 2, 0.515, r"$\mathcal{L}=\mathcal{L}_{\mathrm{Dice}}+\mathcal{L}_{\mathrm{CE}}$", ha="center", va="top", fontsize=10, color=LEARN)
    for i, (k, lab) in enumerate([("sac", "aneurysm sac"), ("parent", "parent artery"), ("branch", "branch arteries"), ("neck", "neck landing zone")]):
        yy0 = 0.445 - i * 0.032
        ax.add_patch(Rectangle((x + 0.03, yy0 - 0.008), 0.018, 0.018, fc=CLASS_COL[k], ec="none", zorder=3))
        ax.text(x + 0.056, yy0, lab, va="center", fontsize=8, color=INK)
    ax.text(x + w / 2, 0.315, "comparators: 3D U-Net, Swin-UNETR", ha="center", fontsize=7.2, color=MUTED)

    # ---------------- geometry
    x, w = cols[2]
    box(ax, x, bot, w, top - bot, COMP_BG, COMP)
    header(ax, x, top - 0.015, w, "2  Morphometry", "and planning factors, computed from the anatomy", COMP)
    ga = fig.add_axes([x + 0.03, 0.58, 0.145, 0.19])
    th = np.linspace(0, 2 * np.pi, 200)
    ga.fill(3.0 * np.cos(th), 2.6 + 2.6 * np.sin(th), color=CLASS_COL["sac"], alpha=0.25, lw=0)
    ga.plot(3.0 * np.cos(th), 2.6 + 2.6 * np.sin(th), color=CLASS_COL["sac"], lw=1.2)
    ga.plot([-2.1, 2.1], [0.75, 0.75], color=CLASS_COL["neck"], lw=3)
    ga.annotate("", xy=(0, 5.6), xytext=(0, 0.75), arrowprops=dict(arrowstyle="-|>", color=COMP, lw=1.3))
    ga.text(0.35, 3.9, r"$\mathbf{n}$", color=COMP, fontsize=11)
    ga.plot([-3.6, -0.8], [-2.6, 0.5], color=CLASS_COL["parent"], lw=5, solid_capstyle="round")
    ga.plot([0.8, 3.6], [0.5, -0.4], color=CLASS_COL["branch"], lw=3.5, solid_capstyle="round")
    ga.plot([-0.5, -3.8], [0.5, 0.0], color=CLASS_COL["branch"], lw=3.5, solid_capstyle="round")
    ga.annotate("", xy=(3.0, 2.6), xytext=(-3.0, 2.6), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
    ga.text(0, 2.85, "W", ha="center", fontsize=8)
    ga.set_xlim(-4.2, 4.2), ga.set_ylim(-3, 6)
    ga.set_aspect("equal")
    ga.axis("off")
    ax.text(x + 0.012, 0.555, "neck plane: PCA of landing-zone voxels,\n" + r"$\mathbf{n}$ = eigenvector of $\lambda_{\min}$ (towards the dome)",
            va="top", fontsize=7.8, color=INK)
    items = ["mean / max width, height, neck, W/N", "volume, solidity, irregularity", "branch incorporation",
             "entry angle (> 60° = angulated)", "daughter asymmetry (> 45°)", "manufacturer's simple-case check"]
    for i, it in enumerate(items):
        ax.text(x + 0.018, 0.48 - i * 0.03, "• " + it, va="top", fontsize=7.8, color=INK)

    # ---------------- simulation
    x, w = cols[3]
    box(ax, x, bot, w, top - bot, COMP_BG, COMP)
    header(ax, x, top - 0.015, w, "3  Virtual deployment", "every catalogue size (29 WEB SL sizes)", COMP)
    ax.text(x + w / 2, 0.80, r"$c=1-\frac{\bar r}{W_d/2},\quad H'=H_d+k\,W_d\,c,\quad g=h_{\mathrm{sac}}-H'$", ha="center", va="top",
            fontsize=10.5, color=COMP)
    ax.text(x + w / 2, 0.735, "conform to the sac wall along the axis; compression c,\ndeployed height H′, dome clearance g, apposition,\nneck coverage and margin, protrusion",
            ha="center", va="top", fontsize=7.8, color=INK)
    ha = fig.add_axes([x + 0.04, 0.395, w - 0.075, 0.215])
    widths = [4, 4.5, 5, 6, 7, 8, 9, 10, 11]
    heights = [2, 3, 4, 5, 6, 7, 8, 9]
    avail = {4: [2, 3], 4.5: [2, 3], 5: [2, 3], 6: [2, 3, 4], 7: [2, 3, 4, 5], 8: [3, 4, 5, 6], 9: [4, 5, 6, 7], 10: [5, 6, 7, 8], 11: [6, 7, 8, 9]}
    G = np.full((len(heights), len(widths)), np.nan)
    for j, wd in enumerate(widths):
        for h in avail[wd]:
            G[heights.index(h), j] = 5.6 - (h + 0.45 * wd * max(0.0, 1 - 6.4 / wd))     # illustrative clearance pattern
    ha.imshow(G, origin="lower", cmap="RdYlGn", vmin=-2, vmax=2, aspect="auto")
    for (i, j) in [(2, 5), (3, 5), (2, 6)]:
        ha.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.4))
    ha.set_xticks(range(len(widths)), [f"{v:g}" for v in widths], fontsize=6.5)
    ha.set_yticks(range(len(heights)), heights, fontsize=6.5)
    ha.set_xlabel("device width (mm)", fontsize=7)
    ha.set_ylabel("height (mm)", fontsize=7)
    ha.set_title("dome clearance g per size (illustrative)", fontsize=7.5, color=MUTED)
    ax.text(x + w / 2, 0.315, "k fitted on development patients with post-deployment VasoCT", ha="center", fontsize=7.2, color=MUTED)

    # ---------------- decision
    x, w = cols[4]
    box(ax, x, bot, w, top - bot, LEARN_BG, LEARN)
    header(ax, x, top - 0.015, w, "4  Size ranking", "learned: conditional logit", LEARN)
    ax.text(x + w / 2, 0.80, r"$P(d\mid x)=\frac{\exp(\beta^{\mathrm{T}}\phi(x,d))}{\sum_{d'\in\mathcal{C}(x)}\exp(\beta^{\mathrm{T}}\phi(x,d'))}$",
            ha="center", va="top", fontsize=10, color=LEARN)
    ax.text(x + 0.012, 0.73, r"$\mathcal{C}(x)=\{d: W_d\geq N_{\max}\}$" + "\n" + r"$\phi$: chart deviation, deployment," + "\nplanning factors, sac image score,\nlocation, rupture, interactions\nL2 penalty by inner CV (by patient)",
            va="top", fontsize=7.8, color=INK)
    ba = fig.add_axes([x + 0.03, 0.39, w - 0.05, 0.13])
    ps = [0.62, 0.21, 0.09]
    ba.barh([2, 1, 0], ps, color=[LEARN, "#93b4f5", "#c7d7fa"], height=0.62)
    for i, (lab, p) in enumerate(zip(["8×4", "8×5", "9×4"], ps)):
        ba.text(-0.02, 2 - i, lab, ha="right", va="center", fontsize=8)
        ba.text(p + 0.02, 2 - i, f"{p:.2f}", va="center", fontsize=7.5, color=MUTED)
    ba.set_xlim(0, 1)
    ba.axis("off")
    ba.set_title("top-3 sizes (example)", fontsize=7.5, color=MUTED)
    ax.text(x + 0.012, 0.37, r"review flag: $\hat p_{(1)}<\tau$" + "\n" + r"$\tau$ = 20th percentile in development CV", va="top", fontsize=7.8, color=INK)

    for (x0, w0), (x1, _) in zip(cols[:-1], cols[1:]):
        arrow(ax, x0 + w0 + 0.002, 0.62, x1 - 0.002, 0.62)

    # ---------------- evaluation band
    box(ax, 0.015, 0.07, 0.97, 0.18, EVAL_BG, EVAL)
    ax.text(0.027, 0.232, "Evaluation", fontsize=11.5, fontweight="bold", color=EVAL, va="top")
    blocks = [
        ("Data partition", "Centres 1 + 2: development (5 patient-level folds)\nCentre 3: independent external validation,\ndesignated before development"),
        ("Sizing outcomes", "exact top-1 vs the implanted WEB size (primary)\ntop-3, within one size, acceptable plan\nmanufacturer's sizing chart as comparator"),
        ("Robustness", "leave-one-centre-out refits (main paired comparison)\nprespecified simple / complex subgroups\nAUROC of the top-1 probability (DeLong CI)"),
        ("Clinical evaluation", "4 readers, unaided then AI-assisted (washout)\nGEE accuracy, mixed-model time, Fleiss' κ,\nNASA-TLX, structured ratings, VasoCT review"),
    ]
    for i, (t, s) in enumerate(blocks):
        bx = 0.03 + i * 0.24
        ax.text(bx, 0.185, t, fontsize=9.2, fontweight="bold", color=INK, va="top")
        ax.text(bx, 0.155, s, fontsize=7.9, color=INK, va="top", linespacing=1.45)

    ax.add_patch(Rectangle((0.70, 0.958), 0.012, 0.012, fc=LEARN_BG, ec=LEARN))
    ax.text(0.716, 0.964, "learned model", va="center", fontsize=8, color=MUTED)
    ax.add_patch(Rectangle((0.79, 0.958), 0.012, 0.012, fc=COMP_BG, ec=COMP))
    ax.text(0.806, 0.964, "geometric computation", va="center", fontsize=8, color=MUTED)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "assets"))
    ap.add_argument("--only", choices=["architecture", "abstract"], default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.only in (None, "architecture"):
        print(architecture(out / "architecture.png"))
    if a.only in (None, "abstract"):
        try:
            from graphical_abstract import graphical_abstract       # noqa: E402  (scripts/graphical_abstract.py)
        except ImportError:
            sys.path.insert(0, str(Path(__file__).parent))
            from graphical_abstract import graphical_abstract       # noqa: E402
        print(graphical_abstract(out / "graphical-abstract.png"))


if __name__ == "__main__":
    main()
