# SPS GenAI Embedding API

This project implements a FastAPI application with spaCy embedding functionality for Assignment 1.

## Features

* `GET /`
  Returns a basic API status message.

* `GET /embedding/{text}`
  Returns vector information for the input text, including whether a vector exists, vector dimension, vector norm, and a preview of the first 10 embedding values.

* `POST /similarity`
  Compares two input texts using spaCy word vectors and returns a semantic similarity score.

## Run Locally

Install dependencies:

```bash
uv sync
```

Start the FastAPI server:

```bash
uv run uvicorn main:app --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

## Run with Docker

Build and start the app:

```bash
docker compose up --build
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

Stop and remove the container:

```bash
docker compose down
```

## Example Requests

Embedding endpoint:

```text
GET /embedding/king
```

Example response includes:

```json
{
  "text": "king",
  "has_vector": true,
  "vector_dimension": 300,
  "vector_norm": 7.141745717143642,
  "embedding_preview": [0.31542, -0.35068, 0.42923]
}
```

Similarity endpoint request body:

```json
{
  "text1": "king",
  "text2": "queen"
}
```

The similarity score for `king` and `queen` should be higher than unrelated words such as `king` and `banana`.

## Repository

https://github.com/RickAnalystCU-web/sps_genai
