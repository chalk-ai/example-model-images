"""
Tweet Toxicity Classifier — model definitions.

Architecture:
  twitter-roberta-base → dropout → classification head → multilabel sigmoid

Supports multilabel toxicity categories:
  toxic, severe_toxic, obscene, threat, insult, identity_hate
"""

import re
from dataclasses import dataclass

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ToxicityConfig:
    model_name: str = "cardiffnlp/twitter-roberta-base"
    max_length: int = 128  # tweets are short, 128 is plenty
    num_labels: int = 6  # multilabel toxicity categories
    label_names: tuple = (
        "toxic",
        "severe_toxic",
        "obscene",
        "threat",
        "insult",
        "identity_hate",
    )
    dropout: float = 0.3
    hidden_dim: int = 768  # roberta-base hidden size
    classifier_hidden: int = 256

    # Training
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    batch_size: int = 32
    epochs: int = 4
    grad_clip: float = 1.0

    # Class weights for imbalanced labels (threats are rare)
    class_weights: tuple = (1.0, 3.0, 1.0, 5.0, 1.0, 3.0)


# ---------------------------------------------------------------------------
# Preprocessing (tweet-specific)
# ---------------------------------------------------------------------------


def preprocess_tweet(text: str) -> str:
    """
    cardiffnlp's model expects a specific preprocessing format:
    - usernames → @user
    - URLs → http
    - keep emojis (the tokenizer handles them)
    """
    # Normalize usernames
    text = re.sub(r"@\w+", "@user", text)
    # Normalize URLs
    text = re.sub(r"https?://\S+", "http", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TweetToxicityClassifier(nn.Module):
    """
    twitter-roberta-base → pooling → MLP → multilabel sigmoid

    Uses mean pooling over non-padded tokens instead of just [CLS],
    which typically performs better for short texts like tweets.
    """

    def __init__(self, cfg: ToxicityConfig):
        super().__init__()
        from transformers import AutoModel

        self.cfg = cfg
        self.roberta = AutoModel.from_pretrained(cfg.model_name)

        # Freeze embeddings + first 6 layers, fine-tune upper 6
        # This stabilises training and prevents catastrophic forgetting
        modules_to_freeze = [
            self.roberta.embeddings,
            *self.roberta.encoder.layer[:6],
        ]
        for module in modules_to_freeze:
            for param in module.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.classifier_hidden),
            nn.GELU(),
            nn.LayerNorm(cfg.classifier_hidden),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.classifier_hidden, cfg.num_labels),
        )

    def mean_pool(
        self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Mean pool over non-padded tokens."""
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)  # (B, H)
        counts = mask_expanded.sum(dim=1).clamp(min=1e-9)  # (B, 1)
        return sum_hidden / counts

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(pooled)  # (B, num_labels)
        return logits

    def predict_proba(self, input_ids, attention_mask) -> torch.Tensor:
        """Return sigmoid probabilities for each toxicity category."""
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            return torch.sigmoid(logits)
