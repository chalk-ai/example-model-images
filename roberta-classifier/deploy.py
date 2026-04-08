"""
Deploy the toxicity model to a Chalk scaling group.

Usage:
    python deploy.py --model-version 1
    python deploy.py --image us-central1-docker.pkg.dev/chalk-infra/public-images/model-examples/roberta-classifier --memory 4Gi --gpu nvidia-tesla-t4:1
"""

import argparse
import os

import pyarrow as pa
from chalk.client import ChalkClient
from chalk.scalinggroup.spec import AutoScalingSpec, ScalingGroupResourceRequest

MODEL_NAME = "TweetToxicityClassifier-server"
SCALING_GROUP_NAME = "tweet-toxicity-classifier"


def deploy(
    image: str | None = None,
    model_version: int | None = None,
    min_replicas: int = 1,
    max_replicas: int = 1,
    cpu: str = "1",
    memory: str = "2Gi",
    gpu: str | None = None,
):
    if image and model_version:
        raise ValueError("Provide either --image or --model-version, not both")
    if not image and not model_version:
        raise ValueError("Provide either --image or --model-version")

    client = ChalkClient(
        client_id=os.getenv("CHALK_CLIENT_ID"),
        client_secret=os.getenv("CHALK_CLIENT_SECRET"),
        api_server=os.getenv("CHALK_API_SERVER"),
    )

    resources = ScalingGroupResourceRequest(
        cpu=cpu,
        memory=memory,
    )
    if gpu:
        resources = ScalingGroupResourceRequest(
            cpu=cpu,
            memory=memory,
            gpu=gpu,
        )

    if image:
        response = client.register_model_version(
            name=MODEL_NAME,
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
            model_image=image,
        )
        version = response.model_version
    else:
        version = model_version

    client.deploy_model_version_to_scaling_group(
        name=SCALING_GROUP_NAME,
        model_name=MODEL_NAME,
        model_version=version,
        scaling=AutoScalingSpec(
            min_replicas=min_replicas,
            max_replicas=max_replicas,
        ),
        resources=resources,
        env_vars={
            "CHALK_CLIENT_ID": os.getenv("CHALK_CLIENT_ID"),
            "CHALK_CLIENT_SECRET": os.getenv("CHALK_CLIENT_SECRET"),
            "CHALK_API_SERVER": os.getenv("CHALK_API_SERVER"),
            "CHALK_INPUT_ARGS": "tweet",
        },
    )
    print(f"Deployed {MODEL_NAME} v{version} to scaling group '{SCALING_GROUP_NAME}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy toxicity model to Chalk")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image",
        type=str,
        help="Docker image URI — registers a new model version and deploys it",
    )
    group.add_argument(
        "--model-version",
        type=int,
        help="Existing model version to deploy (skip registration)",
    )
    parser.add_argument("--min-replicas", type=int, default=1)
    parser.add_argument("--max-replicas", type=int, default=3)
    parser.add_argument("--cpu", type=str, default="1")
    parser.add_argument("--memory", type=str, default="2Gi")
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU spec (e.g. nvidia-tesla-t4:1)",
    )
    args = parser.parse_args()
    deploy(
        image=args.image,
        model_version=args.model_version,
        min_replicas=args.min_replicas,
        max_replicas=args.max_replicas,
        cpu=args.cpu,
        memory=args.memory,
        gpu=args.gpu,
    )
