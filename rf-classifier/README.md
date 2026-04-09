# Random Forest Classifier

Sklearn classifier/regressor. Falls back to a dummy model if no `model.pkl` is provided.

## Build

```bash
docker build --platform linux/amd64 -t rf-classifier .
```

## Run

```bash
docker run -p 8080:8080 rf-classifier
```

To use a pre-trained model, mount or copy `model.pkl` into `/app/`:

```bash
docker run -p 8080:8080 -v ./model.pkl:/app/model.pkl rf-classifier
```

## Handler

- **Input**: `event["features"]` — `pa.Array` of utf8 JSON strings containing feature values
- **Output**: `pa.Array` of utf8 JSON strings with predictions, probabilities, and class labels
