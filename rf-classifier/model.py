"""Random Forest Classifier — chalk-remote-call handler convention.

Example customer model that runs a pre-trained sklearn classifier.
Falls back to a dummy model if no model.pkl is found.
"""

import json

import joblib
import numpy as np
import pyarrow as pa

model = None
feature_names = None


def on_startup():
    """Load the pre-trained model at startup."""
    global model, feature_names
    print("Loading classifier model...")
    try:
        model = joblib.load("/app/model.pkl")
        try:
            feature_names = joblib.load("/app/feature_names.pkl")
        except FileNotFoundError:
            feature_names = None
        print("Model loaded!")
    except FileNotFoundError:
        print("WARNING: model.pkl not found. Using a dummy model for testing.")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier()
        model.fit(np.random.rand(10, 4), np.random.randint(0, 2, 10))


def handler(event: dict[str, pa.Array], context: dict) -> pa.Array:
    """Make predictions using the classifier.

    Parameters
    ----------
    event
        {"features": pa.Array of JSON strings, each a list of floats}

    Returns
    -------
    pa.Array of JSON strings with prediction results
    """
    features_list = event["features"].to_pylist()
    results = []

    for features_json in features_list:
        if features_json is None:
            results.append(None)
            continue

        try:
            # Parse features — could be a JSON string or already a list
            if isinstance(features_json, str):
                features = json.loads(features_json)
            else:
                features = features_json

            X = np.array(features).reshape(1, -1)
            prediction = float(model.predict(X)[0])

            result = {"prediction": prediction}

            # Include probabilities for classifiers
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(X)[0]
                result["probabilities"] = [float(p) for p in probas]
                if hasattr(model, "classes_"):
                    result["class_labels"] = [str(c) for c in model.classes_]

            results.append(json.dumps(result))
        except Exception as e:
            results.append(json.dumps({"error": str(e)}))

    return pa.array(results, type=pa.utf8())
