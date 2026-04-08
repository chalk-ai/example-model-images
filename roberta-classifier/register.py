"""
Register the trained toxicity model in the Chalk model registry.

Usage:
    python register.py
    python register.py --alias v1.0
"""

import argparse
import os

from chalk.client import ChalkClient
from chalk.ml import ModelType, ModelEncoding
import pyarrow as pa


MODEL_NAME = "TweetToxicityClassifier"
MODEL_PATH = "best_toxicity_model.pt"


def register(alias: str | None = None):
    client = ChalkClient(
        client_id=os.getenv("CHALK_CLIENT_ID"),
        client_secret=os.getenv("CHALK_CLIENT_SECRET"),
        api_server=os.getenv("CHALK_API_SERVER"),
    )

    try:
        client.register_model_namespace(
            name=MODEL_NAME,
            description="Multilabel tweet toxicity classifier using cardiffnlp/twitter-roberta-base",
        )
    except Exception as e:
        if "duplicate key value" in str(e):
            print("Model namespace already exists")
        else:
            raise e

    aliases = [alias] if alias else []
    client.register_model_version(
        name=MODEL_NAME,
        aliases=aliases,
        model_paths=[MODEL_PATH],
        model_type=ModelType.PYTORCH,
        model_encoding=ModelEncoding.PICKLE,
        input_schema={"tweet": pa.large_string()},
        output_schema={
            "toxic": pa.float64(),
            "severe_toxic": pa.float64(),
            "obscene": pa.float64(),
            "threat": pa.float64(),
            "insult": pa.float64(),
            "identity_hate": pa.float64(),
            "overall_toxic": pa.bool_(),
        },
        metadata={
            "framework": "pytorch",
            "base_model": "cardiffnlp/twitter-roberta-base",
            "labels": "toxic,severe_toxic,obscene,threat,insult,identity_hate",
        },
    )
    print(f"Registered {MODEL_NAME} from {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register toxicity model in Chalk")
    parser.add_argument(
        "--alias", type=str, default=None, help="Version alias (e.g. v1.0)"
    )
    args = parser.parse_args()
    register(alias=args.alias)
