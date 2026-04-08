# Example Model Images

Example model images using [chalk-remote-call-python](https://pypi.org/project/chalk-remote-call-python/) for Chalk Scaling Groups.

Each model implements a `handler(event, context)` function in `model.py` following the chalk-remote-call convention. The handler receives Arrow Arrays and returns Arrow Arrays.

## Models

### 1. NER Model
**Directory**: `ner-model/`

Named Entity Recognition using spaCy's `en_core_web_sm` model.

```bash
cd ner-model
docker build --platform linux/amd64 -t ner-model .
```

### 2. Random Forest Classifier
**Directory**: `rf-classifier/`

Sklearn classifier/regressor. Falls back to a dummy model if no `model.pkl` is provided.

```bash
cd rf-classifier
docker build --platform linux/amd64 -t rf-classifier .
```

### 3. RoBERTa Tweet Toxicity Classifier
**Directory**: `roberta-classifier/`

Multilabel tweet toxicity classifier using `cardiffnlp/twitter-roberta-base`. Scores tweets across 6 categories: toxic, severe_toxic, obscene, threat, insult, identity_hate.

```bash
cd roberta-classifier
docker build --platform linux/amd64 -t roberta-classifier .
```

**Scripts:**
- `train.py` — train the model and save weights
- `register.py` — register weights in Chalk model registry
- `deploy.py` — register a serving image and deploy to a scaling group
- `test.py` — test the handler locally

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

## Deployment

```bash
# Build any example
cd ner-model-http
docker build --platform linux/amd64 -t my-model:latest .
docker tag my-model:latest ghcr.io/my-org/my-model:latest
docker push ghcr.io/my-org/my-model:latest
```

## Registration

```python
from chalk.client import ChalkClient
from chalk.ml import ModelServingSpec
import pyarrow as pa

client = ChalkClient()
client.register_model_version(
    name="my-model",
    input_schema={"text": pa.large_string()},
    output_schema={"result": pa.large_string()},
    model_image="ghcr.io/my-org/my-model:latest",
    serving=ModelServingSpec(
        min_replicas=1,
        max_replicas=2,
        cpu="2",
        memory="4Gi",
    ),
)
```
