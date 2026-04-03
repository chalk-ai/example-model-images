# NER model (spaCy) using chalk-remote-call-python
#
# Build:
#   docker build --platform linux/amd64 -t ner-model-http .
#
# Run:
#   docker run -p 8080:8080 ner-model-http

FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir chalk-remote-call-python spacy
RUN python -m spacy download en_core_web_sm

COPY model.py /app/model.py

ENV PYTHONPATH=/app

EXPOSE 8080

ENTRYPOINT ["chalk-remote-call", "--handler", "model.handler", "--port", "8080"]
