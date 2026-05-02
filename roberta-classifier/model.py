"""
Serving handler for Tweet Toxicity Classifier.

Follows the chalk-remote-call handler interface:
  on_startup()  — load model + tokenizer once
  handler()     — score a batch of tweets via PyArrow
"""

import logging
import os
import shutil

import torch
import pyarrow as pa
from chalk.client import ChalkClient
from transformers import AutoTokenizer

from classifier import ToxicityConfig, TweetToxicityClassifier, preprocess_tweet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAME = "TweetToxicityClassifier"
MODEL_VERSION = 1

_model: TweetToxicityClassifier | None = None
_tokenizer = None
_cfg: ToxicityConfig | None = None
_device: torch.device | None = None


def on_startup():
    """Load model weights and tokenizer once before the server starts."""
    global _model, _tokenizer, _cfg, _device

    _cfg = ToxicityConfig()
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {_device}")

    _tokenizer = AutoTokenizer.from_pretrained(_cfg.model_name)

    client = ChalkClient(
        client_id=os.getenv("CHALK_CLIENT_ID"),
        client_secret=os.getenv("CHALK_CLIENT_SECRET"),
        api_server=os.getenv("CHALK_API_SERVER"),
    )
    download_dir = os.getenv(
        "MODEL_DIR", os.path.join(os.path.dirname(__file__), "models")
    )
    artifact_path = os.path.join(download_dir, "best_toxicity_model.pt")
    if not os.path.exists(artifact_path):
        logger.info(
            f"Downloading model artifact from {MODEL_NAME} v{MODEL_VERSION} to {artifact_path}"
        )
        model_artifact = client.download_model_artifact(
            name=MODEL_NAME,
            version=MODEL_VERSION,
            download_dir=download_dir,
        )
        shutil.move(model_artifact.downloaded_model_files[0], artifact_path)
    else:
        logger.info(f"Using cached model artifact from {artifact_path}")

    _model = TweetToxicityClassifier(_cfg).to(_device)
    _model.load_state_dict(
        torch.load(artifact_path, map_location=_device, weights_only=True)
    )
    _model.eval()
    logger.info("Model loaded and ready for inference")


def handler(event: dict[str, pa.Array], context: dict) -> pa.Array:
    """Score a batch of tweets for toxicity.

    Input:  event["tweet"] — pa.Array of utf8 tweet strings
    Output: pa.StructArray with one column per label (float64) plus
            `overall_toxic` (bool). The struct fields must match the
            registered output_schema; previously this returned a json-encoded
            VARCHAR which mismatched the catalog-declared struct type and
            tripped a Velox INVALID_STATE assertion downstream.
    """
    tweets = event["tweet"].to_pylist()

    cleaned = [preprocess_tweet(t) for t in tweets]
    enc = _tokenizer(
        cleaned,
        max_length=_cfg.max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(_device)
    attention_mask = enc["attention_mask"].to(_device)

    probs = _model.predict_proba(input_ids, attention_mask)  # (B, num_labels)

    label_names = list(_cfg.label_names)
    label_columns: dict[str, list[float]] = {name: [] for name in label_names}
    overall: list[bool] = []
    for b in range(probs.size(0)):
        per_label = {name: round(probs[b, i].item(), 4) for i, name in enumerate(label_names)}
        for name in label_names:
            label_columns[name].append(per_label[name])
        overall.append(any(per_label[n] > 0.5 for n in label_names))

    field_arrays = [pa.array(label_columns[name], type=pa.float64()) for name in label_names]
    field_arrays.append(pa.array(overall, type=pa.bool_()))
    field_names = label_names + ["overall_toxic"]
    return pa.StructArray.from_arrays(field_arrays, names=field_names)
