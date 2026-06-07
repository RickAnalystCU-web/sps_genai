from fastapi import FastAPI
from pydantic import BaseModel
import spacy

app = FastAPI(
    title="SPS GenAI Embedding API",
    description="A simple FastAPI app that uses spaCy to generate text embeddings and compare semantic similarity.",
    version="1.0.0",
)

nlp = spacy.load("en_core_web_lg")


class SimilarityRequest(BaseModel):
    text1: str
    text2: str


@app.get("/")
def read_root():
    return {
        "message": "SPS GenAI Embedding API is running.",
        "endpoints": {
            "embedding": "/embedding/{text}",
            "similarity": "/similarity",
            "docs": "/docs",
        },
    }


@app.get("/embedding/{text}")
def get_embedding(text: str):
    doc = nlp(text)

    return {
        "text": text,
        "has_vector": bool(doc.has_vector),
        "vector_dimension": len(doc.vector),
        "vector_norm": float(doc.vector_norm),
        "embedding_preview": [float(value) for value in doc.vector[:10]],
    }


@app.post("/similarity")
def get_similarity(request: SimilarityRequest):
    doc1 = nlp(request.text1)
    doc2 = nlp(request.text2)

    return {
        "text1": request.text1,
        "text2": request.text2,
        "text1_has_vector": bool(doc1.has_vector),
        "text2_has_vector": bool(doc2.has_vector),
        "similarity": float(doc1.similarity(doc2)),
    }