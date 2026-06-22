from __future__ import annotations

import json
from pathlib import Path


def load_text_input(input_path: Path | None = None, text: str | None = None, text_column: str = "text"):
    """Load arbitrary text records from --text, CSV, or JSONL input."""
    import pandas as pd

    if text is not None:
        stripped = text.strip()
        if not stripped:
            raise ValueError("--text cannot be empty.")
        return pd.DataFrame([{text_column: stripped}])

    if input_path is None:
        raise ValueError("Provide either --text or --input.")

    input_path = Path(input_path)
    suffix = input_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(input_path)
    elif suffix in {".jsonl", ".ndjson"}:
        rows = []
        with input_path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {line_number} of {input_path}: {exc}") from exc
        frame = pd.DataFrame(rows)
    elif suffix == ".json":
        loaded = json.loads(input_path.read_text(encoding="utf-8"))
        rows = loaded if isinstance(loaded, list) else [loaded]
        frame = pd.DataFrame(rows)
    else:
        raise ValueError("Unsupported input format. Use .csv, .jsonl, .ndjson, .json, or --text.")

    if text_column not in frame.columns:
        available = ", ".join(str(column) for column in frame.columns)
        raise ValueError(f"Text column '{text_column}' not found. Available columns: {available}")

    frame = frame.copy()
    frame[text_column] = frame[text_column].fillna("").astype(str)
    frame = frame[frame[text_column].str.strip() != ""].reset_index(drop=True)
    if len(frame) == 0:
        raise ValueError("No non-empty text rows found to classify.")
    return frame


def prepare_text_frame(frame, text_column: str = "text"):
    """Normalize an arbitrary text frame for the model while preserving original columns."""
    prepared = frame.copy()
    if text_column != "text":
        prepared["text"] = prepared[text_column]
    prepared["source"] = prepared.get("source", "manual_text")
    prepared["title"] = prepared.get("title", "")
    prepared["url"] = prepared.get("url", "")
    prepared["published"] = prepared.get("published", "")
    prepared["source_file"] = prepared.get("source_file", "")
    prepared["word_count"] = prepared["text"].fillna("").astype(str).str.split().str.len()
    return prepared


def apply_uncertainty(predictions, uncertainty_threshold: float | None = None):
    """Add confidence columns and optionally replace low-confidence labels with uncertain."""
    import numpy as np

    results = predictions.copy()
    probability_columns = [column for column in results.columns if column.startswith("ordinal_prob_bit_")]
    if probability_columns:
        probabilities = results[probability_columns].astype(float)
        bit_confidence = np.maximum(probabilities.values, 1 - probabilities.values).min(axis=1)
        results["confidence"] = bit_confidence
    else:
        results["confidence"] = np.nan

    results["model_predicted_bias"] = results["predicted_bias"]
    if uncertainty_threshold is not None:
        low_confidence = results["confidence"] < float(uncertainty_threshold)
        results.loc[low_confidence, "predicted_bias"] = "uncertain"
    return results


def save_text_predictions(predictions, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        predictions.to_csv(output_path, index=False)
    elif suffix in {".jsonl", ".ndjson"}:
        predictions.to_json(output_path, orient="records", lines=True, force_ascii=False)
    elif suffix == ".json":
        predictions.to_json(output_path, orient="records", indent=2, force_ascii=False)
    else:
        raise ValueError("Unsupported output format. Use .csv, .jsonl, .ndjson, or .json.")
