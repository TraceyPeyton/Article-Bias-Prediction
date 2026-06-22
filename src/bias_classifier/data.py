from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .constants import BIAS_ID_TO_TEXT, LABEL_ORDER

TEXT_FIELDS = ["full_text", "content", "text", "article", "body", "article_text", "article_content"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def iter_article_payloads(input_path: Path) -> Iterator[tuple[str, str]]:
    input_path = Path(input_path)
    if input_path.is_dir():
        for path in sorted(input_path.rglob("*.json")):
            yield path.name, path.read_text(encoding="utf-8")
        return

    with zipfile.ZipFile(input_path) as archive:
        for name in archive.namelist():
            if name.endswith(".json"):
                yield name, archive.read(name).decode("utf-8")


def article_text(article: dict[str, Any]) -> str:
    return str(next((article.get(field) for field in TEXT_FIELDS if article.get(field)), ""))


def load_new_articles(input_path: Path):
    import pandas as pd

    rows = []
    seen = set()

    for name, raw_json in iter_article_payloads(Path(input_path)):
        try:
            loaded = json.loads(raw_json)
        except Exception as exc:
            print(f"Skipping {name}: {exc}")
            continue

        articles = loaded if isinstance(loaded, list) else [loaded]
        for article in articles:
            if not isinstance(article, dict):
                continue

            text = article_text(article)
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

            if not text.strip():
                continue

            fingerprint_source = url or f"{source}|{title}|{text[:500]}"
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
                    "text": text,
                    "word_count": len(text.split()),
                    "source_file": name,
                }
            )

    return pd.DataFrame(rows)


def load_split(dataset_dir: Path, split_name: str):
    import pandas as pd

    dataset_dir = Path(dataset_dir)
    json_dir = dataset_dir / "data" / "jsons"
    split_path = dataset_dir / "data" / "splits" / "random" / f"{split_name}.tsv"
    split_df = pd.read_csv(split_path, sep="\t")
    rows = []

    for item in split_df.itertuples(index=False):
        article_path = json_dir / f"{item.ID}.json"
        article = json.loads(article_path.read_text(encoding="utf-8"))
        label_id = int(item.bias)
        text = article.get("content") or article.get("content_original") or article.get("title") or ""

        if isinstance(text, str) and text.strip():
            rows.append(
                {
                    "ID": item.ID,
                    "text": text,
                    "label": label_id,
                    "label_text": BIAS_ID_TO_TEXT[label_id],
                    "title": article.get("title", ""),
                    "source": article.get("source", ""),
                }
            )

    frame = pd.DataFrame(rows)
    if "label_text" in frame:
        frame["label_text"] = pd.Categorical(frame["label_text"], categories=LABEL_ORDER, ordered=True)
    return frame
