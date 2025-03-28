import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union
from embedding_service import EmbeddingService
from model_manager import load_models, load_info, get_info

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    load_info()
    yield

app = FastAPI(
    title="Embedding Microservice",
    description="Generate text embeddings using sentence transformers",
    version="0.1.0",
    lifespan=lifespan
)

embedding_service = EmbeddingService()
logging.basicConfig(level=logging.INFO)

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]] = Field(
        examples=["This is a example string"]
    )
    model: str = Field(
        examples=["all-MiniLM-L6-v2"]
    ) 

class EmbeddingData(BaseModel):
    embedding: List[float]
    index: int
    object: str

class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int

class EmbeddingResponse(BaseModel):
    data: List[EmbeddingData]
    model: str
    usage: Usage
    object: str

class InformationRequest(BaseModel):
    model_name: str

class InformationResponse(BaseModel):
    info: dict[str, str]

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    tokens = 0
    embeddings_response = []
    if isinstance(request.input, str):
        request.input = [request.input]
    try:
        #TODO Create check for available models
        raw_embeddings = embedding_service.generate_embeddings(request.input, request.model)

        for index, single_raw_embedding in enumerate(raw_embeddings):
            tokens += len(request.input[index])
            embeddings_response.append(
                EmbeddingData(embedding=single_raw_embedding, index=index, object="embedding")
            )
        return EmbeddingResponse(
            data=embeddings_response,
            model=request.model,
            usage=Usage(prompt_tokens=tokens, total_tokens=tokens),
            object="list",
        )
        
    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    try:
        return {"status": "healthy"}
    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/info")
async def info(request: InformationRequest) -> InformationResponse:
    try:
        return InformationResponse(get_info(model_name=request.model_name))
    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail=str(e))

#TODO Create info endpoint for model informatio (model name, model link on hugging face, dense vector result size)