from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from typing import List
from rag import process_urls, generate_answer

app = FastAPI(
    title="Real Estate Research API",
    version="1.0.0"
)


class URLRequest(BaseModel):
    urls: List[HttpUrl]


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/process-urls")
def process_urls_api(request: URLRequest):
    if not request.urls:
        raise HTTPException(status_code=400, detail="At least one URL is required")

    statuses = []
    for status in process_urls([str(url) for url in request.urls]):
        statuses.append(status)

    return {
        "message": "URLs processed successfully",
        "steps": statuses
    }


@app.post("/v1/query")
def query_api(request: QueryRequest):
    try:
        answer, sources = generate_answer(request.query)
        return {
            "answer": answer,
            "sources": sources.split("\n") if sources else []
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
