from __future__ import annotations

from collections import Counter
from pathlib import Path

from .constants import LABEL_ORDER


def build_source_report(predictions):
    source_counts = predictions.groupby(["source", "predicted_bias"]).size().unstack(fill_value=0)
    for label in LABEL_ORDER:
        if label not in source_counts.columns:
            source_counts[label] = 0
    source_counts = source_counts[LABEL_ORDER]

    summary = source_counts.copy()
    summary["total_articles"] = summary.sum(axis=1)
    for label in LABEL_ORDER:
        summary[f"{label}_pct"] = summary[label] / summary["total_articles"]
    summary["avg_ordinal_score"] = predictions.groupby("source")["ordinal_score_left_to_right"].mean()
    summary["avg_word_count"] = predictions.groupby("source")["word_count"].mean()
    return summary.sort_values("total_articles", ascending=False).reset_index()


def save_charts(predictions, source_summary, output_dir: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = Counter(predictions["predicted_bias"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(LABEL_ORDER, [counts.get(label, 0) for label in LABEL_ORDER], color=["#4d78b7", "#54a66a", "#d08a35"])
    ax.set_title("Predicted Bias Distribution: New Articles")
    ax.set_ylabel("Articles")
    ax.grid(True, axis="y", alpha=0.25)
    max_count = max(counts.values()) if counts else 0
    for index, label in enumerate(LABEL_ORDER):
        value = counts.get(label, 0)
        ax.text(index, value + max(max_count * 0.02, 0.5), str(value), ha="center")
    fig.tight_layout()
    fig.savefig(output_dir / "overall_bias_distribution.png", dpi=150)
    plt.close(fig)

    top_sources = source_summary.head(12).copy()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    left = np.zeros(len(top_sources))
    for label, color in zip(LABEL_ORDER, ["#4d78b7", "#54a66a", "#d08a35"]):
        values = top_sources[f"{label}_pct"].values
        ax.barh(top_sources["source"], values, left=left, label=label, color=color)
        left += values
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of articles")
    ax.set_title("Predicted Bias Share by Source")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "source_bias_share.png", dpi=150)
    plt.close(fig)


def write_markdown_report(predictions, source_summary, output_dir: Path) -> None:
    output_dir = Path(output_dir)
    overall = predictions["predicted_bias"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    total = len(predictions)
    lines = [
        "# New Articles Source Bias Report",
        "",
        f"Total unique articles classified: **{total}**",
        "",
        "## Overall Prediction Distribution",
        "",
        "| Predicted bias | Articles | Percent |",
        "|---|---:|---:|",
    ]
    for label, count in overall.items():
        lines.append(f"| {label} | {int(count)} | {count / total:.1%} |")

    lines.extend(["", "## Source Summary", "", "| Source | Articles | Left | Center | Right | Avg ordinal score |", "|---|---:|---:|---:|---:|---:|"])
    for row in source_summary.itertuples(index=False):
        lines.append(
            f"| {row.source} | {int(row.total_articles)} | "
            f"{getattr(row, 'left_pct'):.1%} | {getattr(row, 'center_pct'):.1%} | "
            f"{getattr(row, 'right_pct'):.1%} | {row.avg_ordinal_score:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The ordinal score runs from 0=left to 2=right.",
            "- These are model predictions, not ground-truth annotations.",
            "- Source-level summaries should be read as aggregate behavior over this scrape, not permanent labels for outlets.",
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source_bias_report.md").write_text("\n".join(lines), encoding="utf-8")
