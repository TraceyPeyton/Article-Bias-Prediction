from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_new_articles, project_root
from .predict import classify_articles
from .reports import build_source_report, save_charts, write_markdown_report


def classify_new_articles_command(argv: list[str] | None = None) -> int:
    root = project_root()
    parser = argparse.ArgumentParser(description="Classify new articles with the ordinal RoBERTa bias model.")
    parser.add_argument(
        "--input",
        default=str(root / "Article-Bias-Prediction-New-Unstructured"),
        help="Path to a folder of JSON article files or a ZIP containing JSON article files.",
    )
    parser.add_argument("--zip", help="Deprecated alias for --input.")
    parser.add_argument("--model-dir", default=str(root / "bias_ordinal_roberta_model"))
    parser.add_argument("--output-dir", default=str(root / "bias_ordinal_roberta_model" / "new_articles_report"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--skip-charts", action="store_true", help="Write CSV/Markdown outputs without PNG charts.")
    args = parser.parse_args(argv)

    input_path = Path(args.zip or args.input)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    articles = load_new_articles(input_path)
    print(f"Loaded {len(articles)} unique articles from {input_path}")
    if len(articles) == 0:
        raise SystemExit("No article text found to classify.")
    print(articles["source"].value_counts().to_string())

    predictions = classify_articles(articles, model_dir, batch_size=args.batch_size, max_length=args.max_length)
    source_summary = build_source_report(predictions)

    predictions_path = output_dir / "new_articles_ordinal_roberta_predictions.csv"
    summary_path = output_dir / "source_bias_summary.csv"
    predictions.to_csv(predictions_path, index=False)
    source_summary.to_csv(summary_path, index=False)
    if not args.skip_charts:
        save_charts(predictions, source_summary, output_dir)
    write_markdown_report(predictions, source_summary, output_dir)

    print(f"Saved predictions: {predictions_path}")
    print(f"Saved source summary: {summary_path}")
    print(f"Saved report: {output_dir / 'source_bias_report.md'}")
    print("\nOverall predictions:")
    print(predictions["predicted_bias"].value_counts().to_string())
    return 0


def main() -> int:
    return classify_new_articles_command()


if __name__ == "__main__":
    raise SystemExit(main())
