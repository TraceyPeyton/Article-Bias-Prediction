from bias_classifier.constants import LABEL_ORDER
from bias_classifier.ordinal import ordinal_probs_to_labels, to_ordinal_targets


def test_to_ordinal_targets_preserves_notebook_encoding():
    assert to_ordinal_targets([0, 1, 2]) == [
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ]


def test_ordinal_probs_to_labels_decodes_cumulative_bits():
    probabilities = [
        [0.9, 0.1, 0.1],
        [0.9, 0.8, 0.1],
        [0.9, 0.8, 0.7],
    ]
    assert ordinal_probs_to_labels(probabilities) == [0, 1, 2]
    assert LABEL_ORDER == ["left", "center", "right"]
