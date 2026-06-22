from __future__ import annotations

import json
from pathlib import Path


def sigmoid(logits):
    import torch

    return torch.sigmoid(logits)


class OrdinalRobertaClassifierMixin:
    """Marker mixin for the ordinal classifier implementation."""


try:
    import torch
    from torch import nn
    from transformers import AutoConfig, AutoModel
except ImportError:  # pragma: no cover - exercised only when ML deps are absent.
    torch = None
    nn = object
    AutoConfig = None
    AutoModel = None


class OrdinalRobertaClassifier(nn.Module if torch is not None else object):
    def __init__(self, model_name: str, num_ordinal_outputs: int = 3, dropout: float = 0.1, freeze_encoder: bool = False):
        if torch is None:
            raise ImportError("Install the ml dependencies to use OrdinalRobertaClassifier.")
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_ordinal_outputs)
        self.loss_fn = nn.BCEWithLogitsLoss()

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        cls_embedding = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(cls_embedding))

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())

        return {"loss": loss, "logits": logits}


def load_metadata(model_dir: Path) -> dict:
    return json.loads((Path(model_dir) / "ordinal_metadata.json").read_text(encoding="utf-8"))


def load_ordinal_model(model_dir: Path, device=None):
    if torch is None:
        raise ImportError("Install torch, transformers, and safetensors to load the model.")
    from safetensors.torch import load_file

    model_dir = Path(model_dir)
    metadata = load_metadata(model_dir)
    label_order = metadata.get("label_order", ["left", "center", "right"])
    model = OrdinalRobertaClassifier(metadata["base_model"], num_ordinal_outputs=len(label_order))
    state_dict = load_file(model_dir / "model.safetensors")
    model.load_state_dict(state_dict)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, metadata, device
