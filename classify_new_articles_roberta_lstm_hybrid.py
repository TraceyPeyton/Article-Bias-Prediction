import argparse
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


LABEL_ORDER = ["left", "center", "right"]
PROJECT_ROOT = Path(__file__).resolve().parent


class RobertaLSTMOrdinalClassifier(nn.Module):
    def __init__(
        self,
        model_name,
        num_ordinal_outputs=3,
        lstm_hidden_size=128,
        lstm_layers=1,
        dropout=0.25,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.roberta = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_size * 4, num_ordinal_outputs)

    def forward(self, input_ids=None, attention_mask=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_embeddings = outputs.last_hidden_state
        lstm_output, _ = self.lstm(sequence_embeddings)

        mask = attention_mask.unsqueeze(-1).float()
        masked_output = lstm_output * mask
        mean_pool = masked_output.sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        max_pool = masked_output.masked_fill(mask == 0, -1e4).max(dim=1).values
        pooled = torch.cat([mean_pool, max_pool], dim=1)
        return self.classifier(self.dropout(pooled))


def iter_article_payloads(input_path):
    if input_path.is_dir():
        for path in sorted(input_path.rglob("*.json")):
            yield path.name, path.read_text(encoding="utf-8")
        return

    with zipfile.ZipFile(input_path) as archive:
        for name in archive.namelist():
            if name.endswith(".json"):
                yield name, archive.read(name).decode("utf-8")


def load_articles(input_path):
    rows = []
    seen = set()
    text_fields = ["full_text", "content", "text", "article", "body", "article_text", "article_content"]

    for name, raw_json in iter_article_payloads(input_path):
        try:
            loaded = json.loads(raw_json)
        except Exception as exc:
            print(f"Skipping {name}: {exc}")
            continue

        articles = loaded if isinstance(loaded, list) else [loaded]
        for article in articles:
            if not isinstance(article, dict):
                continue

            text = next((article.get(field) for field in text_fields if article.get(field)), "")
            title = article.get("title") or article.get("headline") or ""
            url = article.get("url") or article.get("URL") or article.get("link") or ""
            source = (
                article.get("source")
                or article.get("publisher")
                or article.get("source_name")
                or article.get("outlet")
                or article.get("website")
                or "Unknown"
            )

            if not str(text).strip():
                continue

            fingerprint_source = url or f"{source}|{title}|{str(text)[:500]}"
            fingerprint = hashlib.sha256(fingerprint_source.encode("utf-8", errors="ignore")).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)

            rows.append(
                {
                    "source": str(source).strip() or "Unknown",
                    "title": str(title).strip(),
                    "url": str(url).strip(),
                    "published": article.get("published") or article.get("date") or "",
                    "text": str(text),
                    "word_count": len(str(text).split()),
                    "source_file": name,
                }
            )
    return pd.DataFrame(rows)


def ordinal_probs_to_labels(probabilities, threshold=0.5):
    active_bits = (probabilities >= threshold).astype(np.int64).sum(axis=1)
    return np.clip(active_bits - 1, 0, len(LABEL_ORDER) - 1)


def classify_articles(df, model_dir, batch_size=16, max_length=128):
    metadata = json.loads((model_dir / "hybrid_metadata.json").read_text())
    model_name = metadata["base_model"]
    hparams = metadata["hyperparameters"]

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = RobertaLSTMOrdinalClassifier(
        model_name=model_name,
        num_ordinal_outputs=len(LABEL_ORDER),
        lstm_hidden_size=int(hparams["lstm_hidden_size"]),
        lstm_layers=int(hparams["lstm_layers"]),
        dropout=float(hparams["dropout"]),
    )
    state_dict = load_file(model_dir / "model.safetensors")
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"Using device: {device}")

    all_probs = []
    texts = df["text"].tolist()
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            if (start // batch_size + 1) % 10 == 0:
                print(f"Classified {min(start + batch_size, len(texts))}/{len(texts)}")

    probabilities = np.vstack(all_probs)
    predicted_ids = ordinal_probs_to_labels(probabilities)
    results = df.copy()
    results["predicted_bias_score"] = predicted_ids
    results["predicted_bias"] = [LABEL_ORDER[idx] for idx in predicted_ids]
    results["ordinal_score_left_to_right"] = np.clip(probabilities.sum(axis=1) - 1, 0, 2)
    for idx, label in enumerate(LABEL_ORDER):
        results[f"ordinal_prob_bit_{idx}_{label}"] = probabilities[:, idx]
    return results


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


def save_charts(predictions, source_summary, output_dir):
    counts = Counter(predictions["predicted_bias"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(LABEL_ORDER, [counts.get(label, 0) for label in LABEL_ORDER], color=["#4d78b7", "#54a66a", "#d08a35"])
    ax.set_title("Hybrid Predicted Bias Distribution: New Articles")
    ax.set_ylabel("Articles")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "overall_bias_distribution.png", dpi=150)
    plt.close(fig)

    preferred = ["BBC", "ABC", "CBC", "FOX", "AlJazeera", "NPR"]
    preferred_sources = source_summary[source_summary["source"].isin(preferred)].set_index("source").reindex(preferred)
    other_sources = source_summary[~source_summary["source"].isin(preferred)].sort_values("total_articles", ascending=False)
    source_plot = pd.concat([preferred_sources.dropna(subset=["total_articles"]).reset_index(), other_sources])
    fig, ax = plt.subplots(figsize=(14, 4.8))
    x = np.arange(len(source_plot))
    width = 0.26
    colors = {"center": "#20286b", "left": "#2f67df", "right": "#c4d8f8"}
    for offset, label in zip([-width, 0, width], ["center", "left", "right"]):
        values = source_plot[label].astype(int).values
        bars = ax.bar(x + offset, values, width, label=label.title(), color=colors[label], edgecolor="white")
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, str(int(value)), ha="center", fontsize=8)
    ax.set_title("Hybrid Predicted Bias by News Source")
    ax.set_ylabel("Article Count")
    ax.set_xticks(x, source_plot["source"])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "source_bias_counts.png", dpi=150)
    plt.close(fig)


def write_markdown_report(predictions, source_summary, output_dir):
    overall = predictions["predicted_bias"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    total = len(predictions)
    lines = [
        "# Hybrid RoBERTa-LSTM New Articles Source Bias Report",
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

    lines.extend(
        [
            "",
            "## Source Summary",
            "",
            "| Source | Articles | Left | Center | Right | Avg ordinal score |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
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
            "- These are hybrid RoBERTa-LSTM model predictions, not ground-truth annotations.",
            "- Source-level summaries should be read as aggregate behavior over this scrape, not permanent labels for outlets.",
        ]
    )
    (output_dir / "source_bias_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "Article-Bias-Prediction-New-Unstructured"),
        help="Path to a folder of JSON article files or a ZIP containing JSON article files.",
    )
    parser.add_argument("--zip", help="Deprecated alias for --input.")
    parser.add_argument("--model-dir", default=str(PROJECT_ROOT / "bias_roberta_lstm_hybrid_model" / "final"))
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "bias_roberta_lstm_hybrid_model" / "new_articles_report"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.zip or args.input)
    articles = load_articles(input_path)
    print(f"Loaded {len(articles)} unique articles from {input_path}")
    print(articles["source"].value_counts().to_string())

    predictions = classify_articles(articles, Path(args.model_dir), batch_size=args.batch_size)
    source_summary = build_source_report(predictions)

    predictions.to_csv(output_dir / "new_articles_hybrid_predictions.csv", index=False)
    source_summary.to_csv(output_dir / "source_bias_summary.csv", index=False)
    save_charts(predictions, source_summary, output_dir)
    write_markdown_report(predictions, source_summary, output_dir)

    print(f"Saved predictions: {output_dir / 'new_articles_hybrid_predictions.csv'}")
    print(f"Saved source summary: {output_dir / 'source_bias_summary.csv'}")
    print(f"Saved report: {output_dir / 'source_bias_report.md'}")
    print("\nOverall predictions:")
    print(predictions["predicted_bias"].value_counts().reindex(LABEL_ORDER, fill_value=0).to_string())
    print("\nSource summary:")
    print(source_summary.to_string(index=False))


if __name__ == "__main__":
    main()
