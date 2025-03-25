from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
from .embedding_service import EmbeddingService

app = FastAPI(
    title="Embedding Microservice",
    description="Generate text embeddings using sentence transformers",
    version="0.1.0"
)

embedding_service = EmbeddingService()

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str 

class EmbeddingResponse(BaseModel):
    data: List[List[float]]
    model: str

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    try:
        embeddings = embedding_service.generate_embeddings(request.input, request.model)
        
        return {
            "data": embeddings,
            "model": request.model,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}