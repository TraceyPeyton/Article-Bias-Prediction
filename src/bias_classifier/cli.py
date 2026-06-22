from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_new_articles, project_root
from .predict import classify_articles
from .reports import build_source_report, save_charts, write_markdown_report
from .text_io import apply_uncertainty, load_text_input, prepare_text_frame, save_text_predictions


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


def classify_text_command(argv: list[str] | None = None) -> int:
    root = project_root()
    parser = argparse.ArgumentParser(description="Classify arbitrary text, comments, or social posts with the ordinal bias model.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Single text/comment/post to classify.")
    input_group.add_argument("--input", help="CSV, JSON, JSONL, or NDJSON file containing text rows.")
    parser.add_argument("--text-column", default="text", help="Column/key containing text when using --input.")
    parser.add_argument("--output", help="Output path for batch predictions. Supports .csv, .json, .jsonl, .ndjson.")
    parser.add_argument("--model-dir", default=str(root / "bias_ordinal_roberta_model"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int)
    parser.add_argument(
        "--uncertainty-threshold",
        type=float,
        help="Optional minimum confidence from 0 to 1. Lower-confidence rows are labeled uncertain.",
    )
    args = parser.parse_args(argv)

    if args.uncertainty_threshold is not None and not 0 <= args.uncertainty_threshold <= 1:
        raise SystemExit("--uncertainty-threshold must be between 0 and 1.")

    raw_frame = load_text_input(
        input_path=Path(args.input) if args.input else None,
        text=args.text,
        text_column=args.text_column,
    )
    model_frame = prepare_text_frame(raw_frame, text_column=args.text_column)
    predictions = classify_articles(model_frame, Path(args.model_dir), batch_size=args.batch_size, max_length=args.max_length)
    predictions = apply_uncertainty(predictions, uncertainty_threshold=args.uncertainty_threshold)

    if args.text:
        row = predictions.iloc[0]
        print(f"predicted_bias: {row.predicted_bias}")
        print(f"model_predicted_bias: {row.model_predicted_bias}")
        print(f"confidence: {row.confidence:.3f}")
        print(f"ordinal_score_left_to_right: {row.ordinal_score_left_to_right:.3f}")
        return 0

    output_path = Path(args.output) if args.output else Path(args.input).with_name(f"{Path(args.input).stem}_classified.csv")
    save_text_predictions(predictions, output_path)
    print(f"Classified {len(predictions)} rows")
    print(f"Saved predictions: {output_path}")
    print("\nPredictions:")
    print(predictions["predicted_bias"].value_counts().to_string())
    return 0


def main() -> int:
    return classify_new_articles_command()


def classify_text_main() -> int:
    return classify_text_command()


if __name__ == "__main__":
    raise SystemExit(main())
