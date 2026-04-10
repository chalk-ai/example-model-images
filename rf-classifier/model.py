"""Random Forest Classifier — chalk-remote-call handler convention.

Example customer model that runs a pre-trained sklearn classifier.
Falls back to a dummy model if no model.pkl is found.
"""

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
        {"features": pa.Array of lists of floats}

    Returns
    -------
    pa.StructArray with prediction, probabilities, class_labels, and error fields
    """
    features_list = event["features"].to_pylist()
    predictions = []
    probabilities_list = []
    class_labels_list = []
    errors = []

    for features in features_list:
        if features is None:
            predictions.append(None)
            probabilities_list.append(None)
            class_labels_list.append(None)
            errors.append(None)
            continue

        try:
            X = np.array(features).reshape(1, -1)
            prediction = float(model.predict(X)[0])
            predictions.append(prediction)

            # Include probabilities for classifiers
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(X)[0]
                probabilities_list.append([float(p) for p in probas])
                if hasattr(model, "classes_"):
                    class_labels_list.append([str(c) for c in model.classes_])
                else:
                    class_labels_list.append(None)
            else:
                probabilities_list.append(None)
                class_labels_list.append(None)

            errors.append(None)
        except Exception as e:
            predictions.append(None)
            probabilities_list.append(None)
            class_labels_list.append(None)
            errors.append(str(e))

    # Build struct array
    struct_array = pa.StructArray.from_arrays(
        [
            pa.array(predictions, type=pa.float64()),
            pa.array(probabilities_list, type=pa.list_(pa.float64())),
            pa.array(class_labels_list, type=pa.list_(pa.string())),
            pa.array(errors, type=pa.string()),
        ],
        names=["prediction", "probabilities", "class_labels", "error"],
    )

    return struct_array
