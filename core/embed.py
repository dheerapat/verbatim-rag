"""Shared GTE-ModernBERT embedding helpers (CLS pooling, L2-normalised)."""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


def load_embed_model(model_name, device):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    return model, tokenizer


@torch.no_grad()
def encode(embed_model, texts, batch_size=64, max_length=8192):
    """Return a list of L2-normalised embedding lists, one per text."""
    model, tokenizer = embed_model
    out = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch, max_length=max_length, padding=True, truncation=True,
            return_tensors="pt",
        ).to(model.device)
        cls = model(**inputs).last_hidden_state[:, 0]
        normalized = F.normalize(cls.float(), p=2, dim=1)
        out.extend(e.tolist() for e in normalized)
    return out
