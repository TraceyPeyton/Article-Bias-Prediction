from __future__ import annotations

import random
from pathlib import Path

from .constants import LABEL_ORDER
from .data import load_split
from .model import OrdinalRobertaClassifier
from .ordinal import ordinal_probs_to_labels, to_ordinal_targets


def seed_everything(seed: int = 42) -> None:
    import numpy as np
    import torch
    from transformers import set_seed

    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def stratified_sample(frame, n: int | None, seed: int = 42):
    if n is None or len(frame) <= n:
        return frame.reset_index(drop=True)
    per_class = max(1, n // len(LABEL_ORDER))
    return (
        frame.groupby("label", group_keys=False)
        .apply(lambda group: group.sample(min(len(group), per_class), random_state=seed))
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


class ArticleBiasOrdinalDataset:
    def __init__(self, frame, tokenizer, max_length: int = 128):
        import torch

        self.torch = torch
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ordinal_targets = to_ordinal_targets(self.frame["label"].values)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = self.torch.tensor(self.ordinal_targets[index], dtype=self.torch.float32)
        item["class_label"] = self.torch.tensor(int(row["label"]), dtype=self.torch.long)
        return item


def compute_metrics(eval_pred):
    import numpy as np
    from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error

    logits, labels = eval_pred
    probabilities = 1 / (1 + np.exp(-logits))
    y_pred = ordinal_probs_to_labels(probabilities)
    y_true = ordinal_probs_to_labels(labels)
    y_pred_array = np.asarray(y_pred)
    y_true_array = np.asarray(y_true)

    return {
        "accuracy": accuracy_score(y_true_array, y_pred_array),
        "ordinal_mae": mean_absolute_error(y_true_array, y_pred_array),
        "ordinal_mse": mean_squared_error(y_true_array, y_pred_array),
        "severe_error_rate": float(np.mean(np.abs(y_true_array - y_pred_array) == 2)),
    }


def build_trainer(
    dataset_dir: Path,
    output_dir: Path,
    model_name: str = "distilroberta-base",
    max_length: int = 128,
    seed: int = 42,
    run_full_training: bool = True,
    freeze_encoder: bool = False,
):
    import torch
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    seed_everything(seed)
    max_train_samples = None if run_full_training else 2000
    max_valid_samples = None if run_full_training else 500
    max_test_samples = None if run_full_training else 500

    train_df = stratified_sample(load_split(dataset_dir, "train"), max_train_samples, seed)
    valid_df = stratified_sample(load_split(dataset_dir, "valid"), max_valid_samples, seed)
    test_df = stratified_sample(load_split(dataset_dir, "test"), max_test_samples, seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = ArticleBiasOrdinalDataset(train_df, tokenizer, max_length)
    valid_dataset = ArticleBiasOrdinalDataset(valid_df, tokenizer, max_length)
    test_dataset = ArticleBiasOrdinalDataset(test_df, tokenizer, max_length)
    model = OrdinalRobertaClassifier(model_name, num_ordinal_outputs=len(LABEL_ORDER), freeze_encoder=freeze_encoder)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="no" if not run_full_training else "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=1 if not run_full_training else 3,
        weight_decay=0.01,
        load_best_model_at_end=run_full_training,
        metric_for_best_model="ordinal_mae",
        greater_is_better=False,
        logging_steps=25,
        report_to="none",
        seed=seed,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )
    return trainer, tokenizer, {"train": train_df, "valid": valid_df, "test": test_df}, test_dataset
