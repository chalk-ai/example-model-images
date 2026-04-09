# Example Model Images

Example model images using [chalk-remote-call-python](https://pypi.org/project/chalk-remote-call-python/) for Chalk Scaling Groups.

Each model implements a `handler(event, context)` function in `model.py` following the chalk-remote-call convention. The handler receives Arrow Arrays and returns Arrow Arrays.

## Models

| Model | Directory | Description |
|---|---|---|
| [NER Model](ner-model/) | `ner-model/` | Named Entity Recognition using spaCy |
| [Random Forest Classifier](rf-classifier/) | `rf-classifier/` | Sklearn classifier/regressor |
| [RoBERTa Toxicity Classifier](roberta-classifier/) | `roberta-classifier/` | Tweet toxicity classifier using `cardiffnlp/twitter-roberta-base` |

## Handler Convention

Each `model.py` must define:

```python
import pyarrow as pa

def on_startup():
    """Optional — called once before server starts."""
    # Load model weights, initialize resources, etc.

def handler(event: dict[str, pa.Array], context: dict) -> pa.Array:
    """Process a batch of inputs and return results."""
    inputs = event["input_column"].to_pylist()
    results = [process(x) for x in inputs]
    return pa.array(results, type=pa.utf8())
```

## Image Build

```bash
# Build any example
cd ner-model
docker build --platform linux/amd64 -t my-model:latest .
docker tag my-model:latest ghcr.io/my-org/my-model:latest
docker push ghcr.io/my-org/my-model:latest
```

## Deployment

```python
from chalk.client import ChalkClient
from chalk.scalinggroup.spec import AutoScalingSpec, ScalingGroupResourceRequest
import pyarrow as pa

client = ChalkClient()
response = client.register_model_version(
    name="my-model",
    input_schema={"text": pa.large_string()},
    output_schema={"result": pa.large_string()},
    model_image="ghcr.io/my-org/my-model:latest",
)

client.deploy_model_version_to_scaling_group(
    name="my-scaling-group",
    model_name=response.model_name,
    model_version=response.model_version,
    resources=ScalingGroupResourceRequest(
        cpu="2",
        memory="4Gi"
    ),
    scaling=AutoScalingSpec(
        min_replicas=1,
        max_replicas=2,
    ),
)
```
