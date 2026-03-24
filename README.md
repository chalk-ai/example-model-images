# example-model-images

Examples of model images that you can deploy to Chalk Scaling Groups.

## Overview

This directory contains reference implementations of model servers that can be deployed as Chalk Scaling Groups. Each example demonstrates a different protocol (HTTP/gRPC) and model type.

---

## Models

### 1. NER Model (HTTP)
**Directory**: `ner-model-http/`

Named Entity Recognition (NER) model exposed via FastAPI HTTP endpoints. Uses the spaCy `en_core_web_sm` model to extract named entities from text.

**Protocol**: HTTP (FastAPI)
**Port**: `8000`

#### Endpoints

- **`POST /extract`** - Extract named entities from text
  - **Input**:
    - `text` (string): The text to extract entities from
  - **Output**:
    ```json
    {
      "text": "string",
      "entities": [
        {
          "text": "entity text",
          "label": "PERSON|ORG|GPE|etc",
          "start": 0,
          "end": 5
        }
      ]
    }
    ```

- **`GET /health`** - Health check endpoint
  - **Output**: `{"status": "ok"}`

#### Dependencies

- FastAPI
- Uvicorn
- spaCy (`en_core_web_sm` model)

#### Running Locally

```bash
pip install fastapi uvicorn spacy
python -m spacy download en_core_web_sm
python ner_server.py
# Server runs on http://localhost:8000
```

---

### 2. NER Model (gRPC)
**Directory**: `ner-model-grpc/`

Plain gRPC server implementation of NER using spaCy. This example demonstrates how to deploy a customer's existing gRPC model server that has no Chalk-specific dependencies.

**Protocol**: gRPC
**Port**: `9000`

#### Endpoints

- **`/ner.v1.NerService/Extract`** - Extract named entities from text
  - **Input**: JSON-encoded bytes with schema:
    ```json
    {
      "text": "string"
    }
    ```
  - **Output**: JSON-encoded bytes with schema:
    ```json
    {
      "text": "string",
      "entities": [
        {
          "text": "entity text",
          "label": "PERSON|ORG|GPE|etc",
          "start": 0,
          "end": 5
        }
      ]
    }
    ```

#### Dependencies

- gRPC (grpcio)
- spaCy (`en_core_web_sm` model)

#### Running Locally

```bash
pip install grpcio spacy
python -m spacy download en_core_web_sm
python ner_server_plain_grpc.py
# Server runs on localhost:9000
```

#### Notes

This server is designed to work with the gRPC adapter sidecar, which converts Arrow format (from RemoteCallService) to gRPC calls.

---

### 3. Random Forest Classifier
**Directory**: `rf-classifier/`

Classifier/regressor model exposed via FastAPI. Designed to work with scikit-learn models (Random Forest, etc.) but can be adapted for other model types.

**Protocol**: HTTP (FastAPI)
**Port**: `8000`

#### Endpoints

- **`POST /predict`** - Make a prediction using the model
  - **Input**:
    ```json
    {
      "features": [float, float, ...]
    }
    ```
  - **Output**:
    ```json
    {
      "prediction": 0.5,
      "probabilities": [0.3, 0.7],
      "class_labels": ["class_0", "class_1"]
    }
    ```
  - **Note**: For classifiers, includes `probabilities` and `class_labels` if available

- **`POST /predict_proba`** - Get class probabilities (classifiers only)
  - **Input**:
    ```json
    {
      "features": [float, float, ...]
    }
    ```
  - **Output**:
    ```json
    {
      "probabilities": [0.3, 0.7],
      "class_labels": ["0", "1"]
    }
    ```

- **`GET /health`** - Health check endpoint
  - **Output**: `{"status": "ok"}`

#### Dependencies

- FastAPI
- Uvicorn
- scikit-learn
- joblib
- NumPy

#### Running Locally

```bash
pip install fastapi uvicorn scikit-learn joblib numpy
python classifier_server.py
# Server runs on http://localhost:8000
```

#### Model Loading

The server expects a pre-trained model file at `/app/model.pkl`. Optionally, feature names can be loaded from `/app/feature_names.pkl`. If no model is found, a dummy random forest model is created for testing purposes.

---

## Deployment

Each model includes a Dockerfile for containerization:

```bash
# NER (HTTP)
docker build --platform linux/amd64 -f Dockerfile.ner -t ner-model:latest .

# NER (gRPC)
docker build --platform linux/amd64 -f Dockerfile.ner-plain-grpc -t ner-grpc:latest .

# Classifier
docker build --platform linux/amd64 -f Dockerfile.classifier -t classifier-model:latest .
```

All images include health checks configured for Kubernetes deployments.
