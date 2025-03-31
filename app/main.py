import logging
from typing import Dict, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Union
from embedding_service import EmbeddingService
from model_manager import load_models, load_info, get_info

logging.basicConfig(level=logging.INFO)
embedding_service = EmbeddingService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    logging.info("All models loaded")
    load_info()
    logging.info("All models infos loaded")
    yield

app = FastAPI(
    title="Embedding Microservice",
    description="Generate text embeddings using sentence transformers",
    version="0.1.0",
    lifespan=lifespan
)


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
    model: str

class InformationResponse(BaseModel):
    info: Dict[str, Any]

@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest) -> EmbeddingResponse:
    tokens = 0
    embeddings_response = []
    if isinstance(request.input, str):
        request.input = [request.input]
    try:
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

@app.get("/info/{model}")
async def info(model:str) -> InformationResponse:
    try:
        return InformationResponse(info=get_info(model_name=model))
    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail=str(e))

#TODO Create info endpoint for model informatio (model name, model link on hugging face, dense vector result size)