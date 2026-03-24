"""
Simple FastAPI server to host the NER model for scaling groups
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import spacy

app = FastAPI()

# Load the model once at startup
nlp = None

@app.on_event("startup")
async def startup():
    global nlp
    print("Loading NER model...")
    nlp = spacy.load("en_core_web_sm")
    print("Model loaded!")


class Entity(BaseModel):
    text: str
    label: str
    start: int
    end: int


class NERResponse(BaseModel):
    text: str
    entities: List[Entity]


@app.post("/extract")
async def extract_entities(text: str) -> NERResponse:
    """Extract named entities from text"""
    if not nlp:
        return {"error": "Model not loaded"}

    doc = nlp(text)
    entities = [
        Entity(text=ent.text, label=ent.label_, start=ent.start_char, end=ent.end_char)
        for ent in doc.ents
    ]

    return NERResponse(text=text, entities=entities)


@app.get("/health")
async def health_check():
    """Health check endpoint for Kubernetes"""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
