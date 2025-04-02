from typing import Union, List, Dict, Any
from pydantic import BaseModel


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "input": "This is an example string",
                    "model": "all-MiniLM-L6-v2"
                },
                {
                    "input": ["First example", "Second example"],
                    "model": "all-MiniLM-L6-v2"
                }
            ]
        }
    }

class EmbeddingData(BaseModel):
    embedding: List[float]
    index: int 
    object: str
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "embedding": [0.03, 0.01, -0.02, 0.01],
                    "index": 0,
                    "object": "embedding"
                }
            ]
        }
    }

class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt_tokens": 12,
                    "total_tokens": 12
                }
            ]
        }
    }
    

class EmbeddingResponse(BaseModel):
    data: List[EmbeddingData]
    model: str
    usage: Usage
    object: str
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "data": [
                        {
                            "embedding": [0.03, 0.01, -0.02, 0.01],
                            "index": 0,
                            "object": "embedding"
                        }
                    ],
                    "model": "all-MiniLM-L6-v2",
                    "usage": {
                        "prompt_tokens": 12,
                        "total_tokens": 12
                    },
                    "object": "list"
                }
            ]
        }
    }

class InformationRequest(BaseModel):
    model: str
    model_config = {
        "json_schema_extra": {
            "examples": [
                {"model": "all-MiniLM-L6-v2"}
            ]
        }
    }

class InformationResponse(BaseModel):
    info: Dict[str, Any]
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "info": {
                        "name": "all-MiniLM-L6-v2",
                        "link": "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
                        "dimensions": 384
                    }
                }
            ]
        }
    }