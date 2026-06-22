import pandas as pd
import pytest

from bias_classifier.text_io import apply_uncertainty, load_text_input, prepare_text_frame


def test_load_text_input_from_single_text():
    frame = load_text_input(text=" This is a comment. ")

    assert frame.to_dict(orient="records") == [{"text": "This is a comment."}]


def test_load_text_input_from_csv_with_custom_column(tmp_path):
    csv_path = tmp_path / "comments.csv"
    csv_path.write_text("comment,author\nGood point,Ada\n,Grace\nAnother one,Linus\n", encoding="utf-8")

    frame = load_text_input(input_path=csv_path, text_column="comment")

    assert frame["comment"].tolist() == ["Good point", "Another one"]
    assert frame["author"].tolist() == ["Ada", "Linus"]


def test_load_text_input_errors_when_text_column_missing(tmp_path):
    csv_path = tmp_path / "comments.csv"
    csv_path.write_text("body\nhello\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Text column 'comment' not found"):
        load_text_input(input_path=csv_path, text_column="comment")


def test_prepare_text_frame_preserves_custom_text_column():
    frame = pd.DataFrame([{"comment": "A short post", "source": "reddit"}])

    prepared = prepare_text_frame(frame, text_column="comment")

    assert prepared.loc[0, "text"] == "A short post"
    assert prepared.loc[0, "source"] == "reddit"
    assert prepared.loc[0, "word_count"] == 3


def test_apply_uncertainty_preserves_model_label_and_marks_low_confidence():
    predictions = pd.DataFrame(
        [
            {
                "predicted_bias": "left",
                "ordinal_prob_bit_0_left": 0.95,
                "ordinal_prob_bit_1_center": 0.10,
                "ordinal_prob_bit_2_right": 0.05,
            },
            {
                "predicted_bias": "right",
                "ordinal_prob_bit_0_left": 0.55,
                "ordinal_prob_bit_1_center": 0.51,
                "ordinal_prob_bit_2_right": 0.52,
            },
        ]
    )

    results = apply_uncertainty(predictions, uncertainty_threshold=0.7)

    assert results["model_predicted_bias"].tolist() == ["left", "right"]
    assert results["predicted_bias"].tolist() == ["left", "uncertain"]
    assert results.loc[0, "confidence"] == pytest.approx(0.9)
