# RoBERTa Tweet Toxicity Classifier

Multilabel tweet toxicity classifier using `cardiffnlp/twitter-roberta-base`. Scores tweets across 6 categories: toxic, severe_toxic, obscene, threat, insult, identity_hate.

## Build

```bash
docker build --platform linux/amd64 -t roberta-classifier .
```

Or via Cloud Build:

```bash
gcloud builds submit --config cloudbuild.yaml --project <project> .
```

## Run

```bash
docker run -p 8080:8080 \
  -e CHALK_CLIENT_ID=... \
  -e CHALK_CLIENT_SECRET=... \
  -e CHALK_API_SERVER=... \
  roberta-classifier
```

## Handler

- **Input**: `event["tweet"]` — `pa.Array` of utf8 tweet strings
- **Output**: `pa.Array` of utf8 JSON strings with per-label probabilities and `overall_toxic` flag

## Scripts

| Script | Purpose |
|---|---|
| `train.py` | Train the model and save weights |
| `test.py` | Test the handler locally |
| `register.py` | Register weights in Chalk model registry |
| `deploy.py` | Register a serving image and deploy to a scaling group |

### Train

```bash
python train.py
```

### Register weights

```bash
python register.py --alias v1.0
```

### Deploy

```bash
# New image — register + deploy
python deploy.py --image us-central1-docker.pkg.dev/.../roberta-classifier:latest

# Existing version — deploy only
python deploy.py --model-version 1
```
