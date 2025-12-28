from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, Field
from typing import List
from rag import process_urls, generate_answer

app = FastAPI(
    title="Generic URL Research API (RAG)",
    description=(
        "Production-ready Retrieval-Augmented Generation (RAG) API.\n\n"
        "• Ingests public URLs\n"
        "• Builds a vector knowledge base\n"
        "• Answers questions strictly from provided sources\n\n"
        "Swagger includes default demo values for quick testing."
    ),
    version="1.0.0"
)


# -----------------------------
# Request Schemas (WITH DEFAULTS)
# -----------------------------

class URLRequest(BaseModel):
    urls: List[HttpUrl] = Field(
        default=[
            "https://en.wikipedia.org/wiki/Mortgage"
        ],
        description="Public URLs to ingest into the knowledge base"
    )


class QueryRequest(BaseModel):
    query: str = Field(
        default="Tell payment and debt ratios",
        description="Question to ask based on ingested URLs"
    )


# -----------------------------
# Health Check
# -----------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# -----------------------------
# URL Processing Endpoint
# -----------------------------

@app.post("/v1/process-urls")
def process_urls_api(request: URLRequest):
    if not request.urls:
        raise HTTPException(
            status_code=400,
            detail="At least one URL is required"
        )

    statuses = []
    for status in process_urls([str(url) for url in request.urls]):
        statuses.append(status)

    return {
        "message": "URLs processed successfully",
        "steps": statuses
    }


# -----------------------------
# Query Endpoint
# -----------------------------

@app.post("/v1/query")
def query_api(request: QueryRequest):
    try:
        answer, sources = generate_answer(request.query)

        return {
            "answer": answer,
            "sources": sources.split("\n") if sources else []
        }

    except RuntimeError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
