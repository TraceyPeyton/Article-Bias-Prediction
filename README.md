# Bias Classifier

Bias Classifier trains and applies news-article political-bias classifiers over
left/center/right labels. The training data under `Article-Bias-Prediction-main/`
comes from the Baly et al. Article-Bias-Prediction dataset. The current new-article
inference input is the flattened folder `Article-Bias-Prediction-New-Unstructured/`.

The project also references the Hugging Face political-leaning model/data provenance
at [`matous-volf/political-leaning-politics`](https://huggingface.co/matous-volf/political-leaning-politics).
That model card describes a left/center/right political-leaning classifier based on the
POLITICS model and a combined political-leaning training corpus that includes the
Article Bias Prediction dataset.

## Report and Final Experiments

The project report artifacts are `Bias Classifier Report.docx` and `Bias_Classifier_Report.pdf`.
The presentation artifact is `Bias Classifier Presentation.pptx`. The report is based
on two final experiment notebooks:

- Experiment 1: `bias_lstm_classifier.ipynb`, an LSTM text classifier for
  left/center/right article-bias prediction.
- Experiment 2: `bias_ordinal_roberta_classifier_full_run.ipynb`, an ordinal
  RoBERTa-family transformer classifier that treats political bias as an ordered
  spectrum from left to center to right.

The report lists the project team as Kellin Harris, Lasya Suravajhela, Michael Long,
Prashant Joshi, and Tracey Peyton. Its headline results are 61.4% test accuracy for
the BiLSTM baseline and 71.9% test accuracy for the ordinal DistilRoBERTa model.

Other notebooks and the `bias_roberta_lstm_hybrid_model/` directory are exploratory or
follow-on artifacts. They are useful for comparison, but they are not the two final
experiments named above for the report.

## Current Data Reconciliation

- Raw new-article JSON files: 1,999
- Unique URLs in the current new-article folder: 953
- Rows in the existing ordinal RoBERTa prediction CSV: 944
- Rows in the existing hybrid RoBERTa-LSTM prediction CSV: 944
- Real-world article count described in the report: 1,990 articles from BBC, ABC, CBC,
  FOX, Al Jazeera, and NPR.

The raw file count is larger than the classified count because multiple pull folders
contained duplicate articles and some files contain list-style `all_articles` payloads.
The classifier loaders deduplicate by URL when present, otherwise by source, title, and
the first 500 characters of article text.

The report's 1,990-article framing reflects the experiment corpus used for the written
analysis. The current folder is a later consolidated working copy with 1,999 JSON files,
953 unique URLs, and existing deduplicated prediction CSVs with 944 rows. Rerun the
scripts below to regenerate predictions from the current folder.

## Main Artifacts

- `Article-Bias-Prediction-main/`: original labeled dataset and splits.
- `Article-Bias-Prediction-New-Unstructured/`: current flattened new-article JSON input.
- Hugging Face reference: `matous-volf/political-leaning-politics`, used for
  political-leaning model/data provenance and related left/center/right framing.
- `Bias Classifier Report.docx`: editable final report source.
- `Bias_Classifier_Report.pdf`: exported final report based on Experiment 1 and Experiment 2.
- `Bias Classifier Presentation.pptx`: final presentation deck.
- `bias_lstm_classifier.ipynb`: Experiment 1 final notebook.
- `bias_ordinal_roberta_classifier_full_run.ipynb`: Experiment 2 final notebook.
- `bias_ordinal_roberta_model/`: ordinal DistilRoBERTa classifier, checkpoints, and reports.
- `bias_roberta_lstm_hybrid_model/`: additional DistilRoBERTa plus BiLSTM ordinal classifier and reports.
- `classify_new_articles_ordinal_roberta.py`: runs the ordinal RoBERTa classifier.
- `classify_new_articles_roberta_lstm_hybrid.py`: runs the hybrid classifier.
- `export_new_article_projector_embeddings.py`: exports token embeddings for projector review.
- `FOXPULL.py`: pulls FOX RSS entries. By default it stores RSS descriptions, not full article bodies.

## Reproduce New-Article Predictions

From this folder:

```powershell
python classify_new_articles_ordinal_roberta.py
```

The ordinal RoBERTa script now defaults to `Article-Bias-Prediction-New-Unstructured/`.
It also accepts either a folder or a ZIP:

```powershell
python classify_new_articles_ordinal_roberta.py --input Article-Bias-Prediction-New-Unstructured
```

The additional hybrid model can be rerun separately:

```powershell
python classify_new_articles_roberta_lstm_hybrid.py --input Article-Bias-Prediction-New-Unstructured
```

The old `--zip` argument is still accepted as a deprecated alias for `--input` in both
scripts.

## Interpretation Notes

- Predictions are model outputs, not ground-truth annotations.
- Source-level summaries describe this scrape only and should not be treated as permanent
  labels for outlets.
- FOX records pulled by `FOXPULL.py` may contain RSS descriptions instead of full article
  text, so compare source-level aggregates with care.
