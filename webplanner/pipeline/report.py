"""Planning report: JSON for software, Markdown for the operator."""
from __future__ import annotations

from pathlib import Path
from typing import Union

from ..types import PlanReport
from ..utils.io import write_json

DISCLAIMER = ("Research software output for expert review. Check every recommendation against the anatomy; "
              "not for clinical decision making without qualified human oversight.")


def to_markdown(r: PlanReport) -> str:
    a = r.anatomy
    lines = [f"# WEB planning report: {r.case_id}", "", f"> {DISCLAIMER}", ""]
    lines += ["## Recommended sizes", "", "| Rank | WEB size (mm) | Probability | Compression | Deployed height (mm) | Dome clearance (mm) | Neck coverage |",
              "|---:|:---:|---:|---:|---:|---:|---:|"]
    for i, rec in enumerate(r.top, 1):
        d = rec.deployment
        lines.append(f"| {i} | {rec.size.label} | {rec.probability:.2f} | "
                     + (f"{d.compression:.2f} | {d.deployed_height:.1f} | {d.dome_clearance:.1f} | {d.neck_coverage:.2f} |" if d else "- | - | - | - |"))
    lines += ["", f"Sizing chart: **{r.chart_size.label if r.chart_size else '-'}**. "
              f"Review flag: **{'yes' if r.review_flag else 'no'}** (tau = {r.tau:.2f}).", ""]
    if a is not None:
        m, f = a.morphometry, a.factors
        lines += ["## Anatomy", "", "| Measure | Value |", "|:---|---:|",
                  f"| Mean width (mm) | {m.mean_width:.1f} |", f"| Maximum width (mm) | {m.max_width:.1f} |",
                  f"| Height (mm) | {m.height:.1f} |", f"| Neck width (mm) | {m.neck_width:.1f} |",
                  f"| Width-to-neck ratio | {m.width_neck_ratio:.2f} |", f"| Volume (mm3) | {m.volume_mm3:.0f} |",
                  f"| Solidity | {m.solidity:.2f} |", f"| Branch incorporated | {'yes' if m.branch_incorporated else 'no'} |",
                  f"| Entry angle (deg) | {f.entry_angle_deg:.0f} |", f"| Daughter angle difference (deg) | {f.daughter_angle_diff_deg:.0f} |",
                  f"| Simple case | {'yes' if f.simple_case else 'no'} |", ""]
    if r.notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in r.notes] + [""]
    return "\n".join(lines)


def write_report(r: PlanReport, out_dir: Union[str, Path]) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = r.summary()
    s["top"] = [{**t, "deployment": rec.deployment.features() if rec.deployment else None} for t, rec in zip(s["top"], r.top)]
    s["disclaimer"] = DISCLAIMER
    write_json(s, out / f"{r.case_id}_plan.json")
    (out / f"{r.case_id}_plan.md").write_text(to_markdown(r), encoding="utf-8")
    return out
