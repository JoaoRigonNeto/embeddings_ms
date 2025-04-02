import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from embedding_service import EmbeddingService
from model_manager import load_models, load_info, get_info
from schemas import *

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


@app.post(
        "/v1/embeddings",
        summary="Generate text embeddings",
        description="Accepts raw text and returns corresponding vector embeddings.",
        tags=["Embeddings"]
    )
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


@app.get(
        "/health",
        summary="Health check",
        description="Returns status healthy if server is running",
        tags=["Health"]
    )
async def health_check():
    try:
        return {"status": "healthy"}
    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
        "/info/{model}",
        summary="Get information of models",
        description="Return information on specific models loaded",
        tags=["Information"]
    )
async def info(model:str) -> InformationResponse:
    try:
        return InformationResponse(info=get_info(model_name=model))
    except Exception as e:
        logging.error(e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
        "/available_models",
        summary="Available models",
        description="Returns available models to be used for embeddings",
        tags=["Information"]
    )
async def list_models():
    from model_manager import get_available_models
    return {"models": get_available_models()}
