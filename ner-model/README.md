# NER Model

Named Entity Recognition using spaCy's `en_core_web_sm` model.

## Build

```bash
docker build --platform linux/amd64 -t ner-model .
```

## Run

```bash
docker run -p 8080:8080 ner-model
```

## Handler

- **Input**: `event["text"]` — `pa.Array` of utf8 strings
- **Output**: `pa.Array` of utf8 JSON strings with extracted entities (text, label, start, end)
