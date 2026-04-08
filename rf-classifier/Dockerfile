# Random Forest Classifier using chalk-remote-call-python
#
# Build:
#   docker build --platform linux/amd64 -t rf-classifier .
#
# Run:
#   docker run -p 8080:8080 rf-classifier
#
# To use a pre-trained model, mount or copy model.pkl into /app/:
#   COPY model.pkl /app/model.pkl
#   COPY feature_names.pkl /app/feature_names.pkl  # optional

FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir chalk-remote-call-python scikit-learn joblib numpy

# Uncomment to include a pre-trained model:
# COPY model.pkl /app/model.pkl
# COPY feature_names.pkl /app/feature_names.pkl

COPY model.py /app/model.py

ENV PYTHONPATH=/app

EXPOSE 8080

ENTRYPOINT ["chalk-remote-call", "--handler", "model.handler", "--port", "8080"]
