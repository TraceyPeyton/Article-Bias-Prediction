from __future__ import annotations

from collections.abc import Iterable, Sequence

from .constants import LABEL_ORDER


def to_ordinal_targets(label_ids: Iterable[int], num_classes: int = 3) -> list[list[float]]:
    """Encode class ids as cumulative ordinal targets.

    This preserves the notebook's 3-bit encoding for three labels:
    left=[1,0,0], center=[1,1,0], right=[1,1,1].
    """
    targets: list[list[float]] = []
    for label_id in label_ids:
        label_int = int(label_id)
        targets.append([1.0 if label_int >= threshold else 0.0 for threshold in range(num_classes)])
    return targets


def ordinal_probs_to_labels(probabilities: Sequence[Sequence[float]], threshold: float = 0.5) -> list[int]:
    """Decode cumulative ordinal probabilities into label ids."""
    label_ids: list[int] = []
    max_label = len(LABEL_ORDER) - 1
    for row in probabilities:
        active_bits = sum(1 for probability in row if float(probability) >= threshold)
        label_ids.append(min(max(active_bits - 1, 0), max_label))
    return label_ids
