"""
Training script for Tweet Toxicity Classifier.

Usage:
    python train.py
"""

import logging

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from classifier import ToxicityConfig, TweetToxicityClassifier, preprocess_tweet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class TweetToxicityDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        labels: list[list[float]],
        tokenizer,
        max_length: int = 128,
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = preprocess_tweet(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Weighted BCE Loss (handles class imbalance)
# ---------------------------------------------------------------------------


class WeightedMultilabelLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor):
        super().__init__()
        self.register_buffer("weights", class_weights)

    def forward(self, logits, targets):
        # BCE with logits is numerically more stable than sigmoid + BCE
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weighted = loss * self.weights.unsqueeze(0)
        return weighted.mean()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def get_optimizer_and_scheduler(model, cfg: ToxicityConfig, num_training_steps: int):
    """Separate learning rates: lower for roberta backbone, higher for classifier."""
    param_groups = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "roberta" in n and p.requires_grad
            ],
            "lr": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if "roberta" not in n],
            "lr": cfg.learning_rate * 5,  # classifier head trains faster
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups)

    # Linear warmup then linear decay
    warmup_steps = int(num_training_steps * cfg.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(
            0.0, (num_training_steps - step) / max(1, num_training_steps - warmup_steps)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def train_one_epoch(model, dataloader, optimizer, scheduler, loss_fn, device, cfg):
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device, cfg: ToxicityConfig):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids, attention_mask)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits)
        all_preds.append(probs.cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    # Per-label and overall metrics
    binary_preds = (preds > 0.5).astype(float)
    per_label_acc = {}
    for i, name in enumerate(cfg.label_names):
        correct = (binary_preds[:, i] == labels[:, i]).mean()
        per_label_acc[name] = correct

    try:
        from sklearn.metrics import roc_auc_score, f1_score

        auc_per_label = {}
        for i, name in enumerate(cfg.label_names):
            if labels[:, i].sum() > 0:  # skip if no positives
                auc_per_label[name] = roc_auc_score(labels[:, i], preds[:, i])
        macro_f1 = f1_score(labels, binary_preds, average="macro", zero_division=0)
    except ImportError:
        auc_per_label = {}
        macro_f1 = float("nan")

    return {
        "loss": total_loss / len(dataloader),
        "per_label_accuracy": per_label_acc,
        "per_label_auc": auc_per_label,
        "macro_f1": macro_f1,
    }


# ---------------------------------------------------------------------------
# End-to-end Training
# ---------------------------------------------------------------------------


def main():
    from transformers import AutoTokenizer

    cfg = ToxicityConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = TweetToxicityClassifier(cfg).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Parameters: {total:,} total, {trainable:,} trainable ({100 * trainable / total:.1f}%)"
    )

    # ------------------------------------------------------------------
    # Synthetic training data (replace with real data, e.g. Jigsaw dataset)
    # ------------------------------------------------------------------
    import random

    random.seed(42)

    toxic_templates = [
        "you're such a {slur}, go {threat} yourself @user",
        "@user shut up you stupid {insult} nobody asked",
        "i hope you {threat} you worthless piece of garbage",
        "@user you're disgusting and everyone hates you",
        "all {group} are the same, they should be {threat}",
        "@user die in a fire you absolute moron",
        "what a braindead take, you {insult} clown",
        "@user ratio + L + nobody cares + you're trash",
    ]

    clean_templates = [
        "just had the best coffee this morning! great start to the day",
        "@user congrats on the new job! so happy for you",
        "the sunset tonight is absolutely beautiful wow",
        "anyone watching the game tonight? should be a good one",
        "really enjoyed that new album, the production is incredible",
        "@user thanks for sharing, this is really helpful!",
        "cooking pasta for dinner, nothing beats a simple meal",
        "just finished a 5k run, feeling great about the progress",
    ]

    samples_text, samples_labels = [], []
    for _ in range(400):
        if random.random() < 0.35:
            text = random.choice(toxic_templates)
            # multilabel: [toxic, severe_toxic, obscene, threat, insult, identity_hate]
            labels = [
                1.0,
                float(random.random() < 0.3),
                float(random.random() < 0.6),
                float("threat" in text or random.random() < 0.2),
                float(random.random() < 0.7),
                float("group" in text or random.random() < 0.15),
            ]
        else:
            text = random.choice(clean_templates)
            labels = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        samples_text.append(text)
        samples_labels.append(labels)

    # Split 80/20
    split = int(0.8 * len(samples_text))
    train_ds = TweetToxicityDataset(
        samples_text[:split], samples_labels[:split], tokenizer, cfg.max_length
    )
    val_ds = TweetToxicityDataset(
        samples_text[split:], samples_labels[split:], tokenizer, cfg.max_length
    )

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size * 2)

    # Loss + optimizer
    class_weights = torch.tensor(cfg.class_weights, dtype=torch.float32).to(device)
    loss_fn = WeightedMultilabelLoss(class_weights)

    num_training_steps = len(train_dl) * cfg.epochs
    optimizer, scheduler = get_optimizer_and_scheduler(model, cfg, num_training_steps)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    best_f1 = 0.0
    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(
            model, train_dl, optimizer, scheduler, loss_fn, device, cfg
        )
        val_metrics = evaluate(model, val_dl, loss_fn, device, cfg)

        logger.info(
            f"Epoch {epoch + 1}/{cfg.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"macro_f1={val_metrics['macro_f1']:.3f}"
        )
        for name, auc in val_metrics["per_label_auc"].items():
            logger.info(
                f"  {name:15s}  AUC={auc:.3f}  acc={val_metrics['per_label_accuracy'][name]:.3f}"
            )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(model.state_dict(), "models/best_toxicity_model.pt")
            logger.info(f"  → saved best model (f1={best_f1:.3f})")


if __name__ == "__main__":
    main()
