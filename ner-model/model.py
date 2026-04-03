"""NER model using spaCy — chalk-remote-call handler convention.

Example customer model that extracts named entities from text.
"""

import json

import pyarrow as pa
import spacy

nlp = None


def on_startup():
    """Load the spaCy model once at startup."""
    global nlp
    print("Loading NER model...")
    nlp = spacy.load("en_core_web_sm")
    print("Model loaded!")


def handler(event: dict[str, pa.Array], context: dict) -> pa.Array:
    """Extract named entities from text.

    Parameters
    ----------
    event
        {"text": pa.Array of strings}

    Returns
    -------
    pa.Array of JSON strings with entity results
    """
    texts = event["text"].to_pylist()
    results = []

    for text in texts:
        if text is None:
            results.append(None)
            continue

        doc = nlp(text)
        entities = [
            {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
            for ent in doc.ents
        ]

        results.append(json.dumps({"text": text, "entities": entities}))

    return pa.array(results, type=pa.utf8())
