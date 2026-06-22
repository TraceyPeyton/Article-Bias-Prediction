from __future__ import annotations

from pathlib import Path

from .constants import LABEL_ORDER
from .model import load_ordinal_model
from .ordinal import ordinal_probs_to_labels


def classify_articles(frame, model_dir: Path, batch_size: int = 32, max_length: int | None = None):
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    model, metadata, device = load_ordinal_model(Path(model_dir))
    max_length = max_length or metadata.get("crash_safe_defaults", {}).get("max_length", 128)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    label_order = metadata.get("label_order", LABEL_ORDER)

    all_probs = []
    texts = frame["text"].tolist()
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
            logits = model(**encoded)["logits"]
            all_probs.append(torch.sigmoid(logits).cpu().numpy())

    probabilities = np.vstack(all_probs)
    predicted_ids = ordinal_probs_to_labels(probabilities)
    results = frame.copy()
    results["predicted_bias_score"] = predicted_ids
    results["predicted_bias"] = [label_order[index] for index in predicted_ids]
    results["ordinal_score_left_to_right"] = np.clip(probabilities.sum(axis=1) - 1, 0, len(label_order) - 1)
    for index, label in enumerate(label_order):
        results[f"ordinal_prob_bit_{index}_{label}"] = probabilities[:, index]
    return results


def predict_text(text: str, model_dir: Path, max_length: int | None = None) -> dict:
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    model, metadata, device = load_ordinal_model(Path(model_dir))
    max_length = max_length or metadata.get("crash_safe_defaults", {}).get("max_length", 128)
    label_order = metadata.get("label_order", LABEL_ORDER)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    encoded = tokenizer(text, truncation=True, max_length=max_length, padding="max_length", return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded)["logits"]
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    predicted_id = ordinal_probs_to_labels([probs])[0]
    return {
        "predicted_bias": label_order[predicted_id],
        "predicted_id": int(predicted_id),
        "ordinal_score_left_to_right": float(np.clip(probs.sum() - 1, 0, len(label_order) - 1)),
        "cumulative_probabilities": {f"bit_{index}": float(value) for index, value in enumerate(probs)},
    }
