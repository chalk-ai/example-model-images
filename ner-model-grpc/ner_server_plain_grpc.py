"""
Plain gRPC NER server — no Chalk dependencies.

This is an example of a customer's model server that exposes a gRPC endpoint.
It uses standard grpcio and accepts/returns JSON-encoded bytes.

The gRPC adapter sidecar converts Arrow (RemoteCallService) → gRPC (this server).

Usage:
    pip install grpcio spacy
    python -m spacy download en_core_web_sm
    python ner_server_plain_grpc.py
"""

import json
from concurrent import futures

import grpc
import spacy

nlp = None


def extract_entities(request_bytes, context):
    """Handle a gRPC call: receives JSON bytes, returns JSON bytes."""
    global nlp
    if nlp is None:
        print("Loading NER model...")
        nlp = spacy.load("en_core_web_sm")
        print("Model loaded!")

    payload = json.loads(request_bytes)
    text = payload.get("text", "")

    doc = nlp(text)
    entities = [
        {"text": ent.text, "label": ent.label_, "start": ent.start_char, "end": ent.end_char}
        for ent in doc.ents
    ]

    response = {"text": text, "entities": entities}
    return json.dumps(response).encode("utf-8")


class NerServiceServicer:
    """Generic gRPC servicer that handles raw bytes."""
    pass


class NerGenericHandler(grpc.GenericRpcHandler):
    """Routes /ner.v1.NerService/Extract to the extract_entities function."""

    def service(self, handler_call_details):
        if handler_call_details.method == "/ner.v1.NerService/Extract":
            return grpc.unary_unary_rpc_method_handler(
                extract_entities,
                request_deserializer=None,
                response_serializer=None,
            )
        return None


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    server.add_generic_rpc_handlers([NerGenericHandler()])
    server.add_insecure_port("0.0.0.0:9000")
    server.start()
    print("Plain gRPC NER server ready on port 9000")
    print("  Method: /ner.v1.NerService/Extract")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
