from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bar_svg(path: Path, rows: list[dict[str, object]]) -> None:
    width = 760
    height = 360
    left = 190
    top = 50
    bar_h = 22
    gap = 20
    max_value = max(float(row["after"]) for row in rows) if rows else 1.0
    max_value = max(max_value, max(float(row["before"]) for row in rows), 0.001)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="30" font-family="Arial, sans-serif" font-size="18" font-weight="700">Style Relabel v1: Before vs After</text>',
        '<text x="24" y="52" font-family="Arial, sans-serif" font-size="12" fill="#555">120 hard cases, extended40 axes</text>',
    ]
    for i, row in enumerate(rows):
        y = top + 38 + i * (bar_h * 2 + gap)
        before = float(row["before"])
        after = float(row["after"])
        bw = int((before / max_value) * 470)
        aw = int((after / max_value) * 470)
        label = str(row["metric"])
        lines.extend(
            [
                f'<text x="24" y="{y + 16}" font-family="Arial, sans-serif" font-size="12" fill="#222">{label}</text>',
                f'<rect x="{left}" y="{y}" width="{bw}" height="{bar_h}" rx="3" fill="#b9c0ca"/>',
                f'<rect x="{left}" y="{y + bar_h + 4}" width="{aw}" height="{bar_h}" rx="3" fill="#2f6fed"/>',
                f'<text x="{left + bw + 8}" y="{y + 16}" font-family="Arial, sans-serif" font-size="12" fill="#555">{before:.4f}</text>',
                f'<text x="{left + aw + 8}" y="{y + bar_h + 20}" font-family="Arial, sans-serif" font-size="12" fill="#1f4fb3">{after:.4f}</text>',
            ]
        )
    lines.extend(
        [
            '<rect x="560" y="54" width="14" height="14" fill="#b9c0ca"/>',
            '<text x="580" y="66" font-family="Arial, sans-serif" font-size="12" fill="#555">before</text>',
            '<rect x="640" y="54" width="14" height="14" fill="#2f6fed"/>',
            '<text x="660" y="66" font-family="Arial, sans-serif" font-size="12" fill="#555">after</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_package(root: Path, output_dir: Path) -> dict[str, object]:
    relabel_dir = root / "outputs" / "research" / "style_relabel_v1"
    bias_dir = root / "outputs" / "research" / "style_bias_audit"
    output_dir.mkdir(parents=True, exist_ok=True)

    apply_manifest = _read_json(relabel_dir / "style_relabel_apply_manifest.json")
    candidate_manifest = _read_json(relabel_dir / "style_relabel_candidates_manifest.json")
    before_audit = _read_json(bias_dir / "style_bias_audit.json")
    after_audit = _read_json(relabel_dir / "style_bias_audit_after_relabel.json")
    before_keep = before_audit["keep_summaries"][0]
    after_keep = after_audit["keep_summaries"][0]

    summary_rows = [
        {
            "scope": "relabel_subset",
            "metric": "negative_raw_mean",
            "before": apply_manifest["base_relabel_subset_negative_raw_mean"],
            "after": apply_manifest["output_relabel_subset_negative_raw_mean"],
            "delta": round(
                apply_manifest["output_relabel_subset_negative_raw_mean"]
                - apply_manifest["base_relabel_subset_negative_raw_mean"],
                6,
            ),
        },
        {
            "scope": "relabel_subset",
            "metric": "soft_bias_mean",
            "before": apply_manifest["base_relabel_subset_soft_bias_mean"],
            "after": apply_manifest["output_relabel_subset_soft_bias_mean"],
            "delta": round(
                apply_manifest["output_relabel_subset_soft_bias_mean"]
                - apply_manifest["base_relabel_subset_soft_bias_mean"],
                6,
            ),
        },
        {
            "scope": "keep_rows",
            "metric": "negative_raw_mean",
            "before": before_keep["negative_raw_mean"],
            "after": after_keep["negative_raw_mean"],
            "delta": round(after_keep["negative_raw_mean"] - before_keep["negative_raw_mean"], 6),
        },
        {
            "scope": "keep_rows",
            "metric": "soft_bias_mean",
            "before": before_keep["soft_bias_mean"],
            "after": after_keep["soft_bias_mean"],
            "delta": round(after_keep["soft_bias_mean"] - before_keep["soft_bias_mean"], 6),
        },
        {
            "scope": "keep_rows",
            "metric": "edge_mean",
            "before": before_keep["edge_mean"],
            "after": after_keep["edge_mean"],
            "delta": round(after_keep["edge_mean"] - before_keep["edge_mean"], 6),
        },
    ]
    _write_csv(output_dir / "summary_metrics.csv", summary_rows, ["scope", "metric", "before", "after", "delta"])

    decoder_rows = [
        {"target_set": "original", "axis_group": "all", "mae": 0.120546, "target_mean": 0.393866},
        {"target_set": "original", "axis_group": "soft", "mae": 0.118468, "target_mean": 0.814817},
        {"target_set": "original", "axis_group": "negative_raw", "mae": 0.007217, "target_mean": 0.003155},
        {"target_set": "original", "axis_group": "edge", "mae": 0.132776, "target_mean": 0.103873},
        {"target_set": "style_relabel_v1", "axis_group": "all", "mae": 0.136223, "target_mean": 0.393768},
        {"target_set": "style_relabel_v1", "axis_group": "soft", "mae": 0.150443, "target_mean": 0.784605},
        {"target_set": "style_relabel_v1", "axis_group": "negative_raw", "mae": 0.058784, "target_mean": 0.025772},
        {"target_set": "style_relabel_v1", "axis_group": "edge", "mae": 0.155046, "target_mean": 0.124054},
    ]
    _write_csv(output_dir / "decoder_group_mae.csv", decoder_rows, ["target_set", "axis_group", "mae", "target_mean"])

    focus_rows = []
    before_focus = before_keep["focus_axes"]
    after_focus = after_keep["focus_axes"]
    for axis in sorted(set(before_focus) | set(after_focus)):
        before = float(before_focus.get(axis, 0.0))
        after = float(after_focus.get(axis, 0.0))
        focus_rows.append({"axis": axis, "before": before, "after": after, "delta": round(after - before, 6)})
    _write_csv(output_dir / "focus_axes_before_after.csv", focus_rows, ["axis", "before", "after", "delta"])

    chart_rows = [
        {"metric": "relabel subset negative raw", "before": summary_rows[0]["before"], "after": summary_rows[0]["after"]},
        {"metric": "relabel subset soft bias", "before": summary_rows[1]["before"], "after": summary_rows[1]["after"]},
        {"metric": "all kept negative raw", "before": summary_rows[2]["before"], "after": summary_rows[2]["after"]},
        {"metric": "all kept soft bias", "before": summary_rows[3]["before"], "after": summary_rows[3]["after"]},
    ]
    _bar_svg(output_dir / "style_relabel_before_after.svg", chart_rows)

    tex = r"""\begin{table}[t]
\centering
\caption{Effect of felt-state/style relabeling on style supervision bias.}
\begin{tabular}{llrrr}
\toprule
Scope & Metric & Before & After & $\Delta$ \\
\midrule
Relabel subset & Negative raw mean & 0.0010 & 0.3247 & +0.3236 \\
Relabel subset & Soft bias mean & 0.8472 & 0.4149 & -0.4323 \\
All kept rows & Negative raw mean & 0.0032 & 0.0258 & +0.0226 \\
All kept rows & Soft bias mean & 0.8148 & 0.7846 & -0.0302 \\
All kept rows & Edge mean & 0.1039 & 0.1241 & +0.0202 \\
\bottomrule
\end{tabular}
\label{tab:style-relabel-v1}
\end{table}
"""
    (output_dir / "paper_table_style_relabel_v1.tex").write_text(tex, encoding="utf-8")

    readme = f"""# Style Relabel v1 Artifact Package

This package summarizes the current style-bias mitigation proof of concept.

## Key Files

- `summary_metrics.csv`: before/after bias metrics.
- `focus_axes_before_after.csv`: axis-level means for key style and raw-affect axes.
- `decoder_group_mae.csv`: decoder MAE by axis group.
- `style_relabel_before_after.svg`: compact figure for slides or paper draft.
- `paper_table_style_relabel_v1.tex`: LaTeX table.
- `STYLE_RELABEL_V1_REPORT.md`: short narrative report.

## Source Run

- Candidate rows: `{candidate_manifest["rows"]}`
- Applied relabel rows: `{apply_manifest["applied_rows"]}`
- Candidate buckets: `{json.dumps(candidate_manifest["bucket_counts"], ensure_ascii=False)}`

## Main Result

Relabeled subset negative raw mean changed from `{apply_manifest["base_relabel_subset_negative_raw_mean"]}` to `{apply_manifest["output_relabel_subset_negative_raw_mean"]}`.
Relabeled subset soft bias mean changed from `{apply_manifest["base_relabel_subset_soft_bias_mean"]}` to `{apply_manifest["output_relabel_subset_soft_bias_mean"]}`.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    shutil.copyfile(relabel_dir / "STYLE_RELABEL_V1_REPORT.md", output_dir / "STYLE_RELABEL_V1_REPORT.md")
    return {
        "output_dir": str(output_dir),
        "files": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact artifact package for style relabel v1.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="outputs/research/style_relabel_v1/artifact_package")
    args = parser.parse_args()
    payload = build_package(Path(args.root), Path(args.output_dir))
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
