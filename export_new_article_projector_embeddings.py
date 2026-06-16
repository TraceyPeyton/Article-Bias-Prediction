import argparse
from collections import Counter
from pathlib import Path

import pandas as pd
from safetensors.torch import load_file
from transformers import AutoTokenizer


DEFAULT_SUBSETS = {
    "left": ("predicted_bias", "left"),
    "right": ("predicted_bias", "right"),
    "fox": ("source", "FOX"),
    "npr": ("source", "NPR"),
}
PROJECT_ROOT = Path(__file__).resolve().parent


def normalize_token(token):
    return token.replace("Ġ", "").replace("Ċ", "\\n").strip()


def collect_token_counts(frame, tokenizer, max_length):
    special_ids = set(tokenizer.all_special_ids)
    counts = Counter()

    for text in frame["text"].fillna("").astype(str):
        token_ids = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
        )["input_ids"]
        counts.update(token_id for token_id in token_ids if token_id not in special_ids)

    return counts


def write_projector_files(subset_name, subset_frame, token_counts, embedding_matrix, tokenizer, output_dir, max_tokens):
    output_dir.mkdir(parents=True, exist_ok=True)
    top_tokens = token_counts.most_common(max_tokens)

    embeddings_path = output_dir / f"{subset_name}_embeddings.tsv"
    metadata_path = output_dir / f"{subset_name}_metadata.tsv"

    with embeddings_path.open("w", encoding="utf-8") as emb_f, metadata_path.open("w", encoding="utf-8") as meta_f:
        meta_f.write("token\ttoken_text\tcount\tsubset\tarticles\tsources\tpredicted_biases\n")
        source_summary = ",".join(sorted(subset_frame["source"].dropna().astype(str).unique()))
        bias_summary = ",".join(sorted(subset_frame["predicted_bias"].dropna().astype(str).unique()))
        article_count = len(subset_frame)

        for token_id, count in top_tokens:
            if token_id >= embedding_matrix.shape[0]:
                continue
            vector = embedding_matrix[token_id].tolist()
            token = tokenizer.convert_ids_to_tokens(int(token_id))
            token_text = normalize_token(token)
            if not token_text:
                continue

            emb_f.write("\t".join(f"{value:.8g}" for value in vector) + "\n")
            safe_token = token.replace("\t", " ").replace("\n", "\\n")
            safe_text = token_text.replace("\t", " ").replace("\n", "\\n")
            meta_f.write(
                f"{safe_token}\t{safe_text}\t{count}\t{subset_name}\t"
                f"{article_count}\t{source_summary}\t{bias_summary}\n"
            )

    return embeddings_path, metadata_path, len(top_tokens)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default=str(
            PROJECT_ROOT
            / "bias_ordinal_roberta_model"
            / "new_articles_report"
            / "new_articles_ordinal_roberta_predictions.csv"
        ),
    )
    parser.add_argument("--model-dir", default=str(PROJECT_ROOT / "bias_ordinal_roberta_model"))
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "bias_ordinal_roberta_model" / "new_articles_projector_embeddings"),
    )
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=2000)
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    predictions = pd.read_csv(predictions_path)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    state_dict = load_file(model_dir / "model.safetensors")
    embedding_matrix = state_dict["encoder.embeddings.word_embeddings.weight"].cpu()

    manifest_rows = []
    for subset_name, (column, value) in DEFAULT_SUBSETS.items():
        subset_frame = predictions[predictions[column].astype(str).str.lower() == value.lower()].copy()
        if subset_frame.empty:
            print(f"Skipping {subset_name}: no rows where {column}={value}")
            continue

        token_counts = collect_token_counts(subset_frame, tokenizer, args.max_length)
        embeddings_path, metadata_path, token_count = write_projector_files(
            subset_name=subset_name,
            subset_frame=subset_frame,
            token_counts=token_counts,
            embedding_matrix=embedding_matrix,
            tokenizer=tokenizer,
            output_dir=output_dir,
            max_tokens=args.max_tokens,
        )
        manifest_rows.append(
            {
                "subset": subset_name,
                "filter_column": column,
                "filter_value": value,
                "articles": len(subset_frame),
                "unique_tokens_seen": len(token_counts),
                "tokens_exported": token_count,
                "embeddings_file": embeddings_path.name,
                "metadata_file": metadata_path.name,
            }
        )
        print(
            f"{subset_name}: {len(subset_frame)} articles, "
            f"{len(token_counts)} unique tokens, exported {token_count}"
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
