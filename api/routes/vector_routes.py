from fastapi import APIRouter, Depends, HTTPException, Query, Body
from typing import List, Optional
from pydantic import BaseModel
from ..middleware.auth import get_current_user
from ..controllers import vector_controller
from ..schemas.vector_schemas import (
    VectorUpsertRequest,
    VectorSearchRequest,
    VectorSearchResponse,
    VectorMetadata
)

router = APIRouter(prefix="/vectors", tags=["vectors"])

class BatchVectorRequest(BaseModel):
    vectors: List[VectorUpsertRequest]

@router.post("/upsert", response_model=VectorSearchResponse)
async def upsert_vector(
    request: VectorUpsertRequest,
    current_user = Depends(get_current_user)
):
    """
    Create or update vector embeddings for a document or chunk.
    """
    try:
        result = await vector_controller.upsert_vector(
            vector_data=request,
            user_id=current_user.id
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/batch-upsert", response_model=List[VectorSearchResponse])
async def batch_upsert_vectors(
    request: BatchVectorRequest,
    current_user = Depends(get_current_user)
):
    """
    Batch create or update vector embeddings.
    """
    try:
        results = await vector_controller.batch_upsert_vectors(
            vector_data_list=request.vectors,
            user_id=current_user.id
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/search", response_model=List[VectorSearchResponse])
async def search_vectors(
    request: VectorSearchRequest,
    current_user = Depends(get_current_user)
):
    """
    Search for similar vectors using cosine similarity.
    """
    try:
        results = await vector_controller.search_vectors(
            query_vector=request.query_vector,
            top_k=request.top_k,
            threshold=request.threshold,
            filters=request.filters,
            user_id=current_user.id
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/{vector_id}", response_model=VectorSearchResponse)
async def get_vector(
    vector_id: str,
    current_user = Depends(get_current_user)
):
    """
    Retrieve a specific vector by ID.
    """
    vector = await vector_controller.get_vector(
        vector_id=vector_id,
        user_id=current_user.id
    )
    if not vector:
        raise HTTPException(status_code=404, detail="Vector not found")
    return vector

@router.delete("/{vector_id}")
async def delete_vector(
    vector_id: str,
    current_user = Depends(get_current_user)
):
    """
    Delete a specific vector.
    """
    success = await vector_controller.delete_vector(
        vector_id=vector_id,
        user_id=current_user.id
    )
    if not success:
        raise HTTPException(status_code=404, detail="Vector not found")
    return {"message": "Vector deleted successfully"}

@router.post("/reindex")
async def reindex_vectors(
    document_ids: List[str] = Body(...),
    model_name: Optional[str] = Body(None),
    current_user = Depends(get_current_user)
):
    """
    Reindex vectors for specified documents, optionally with a new model.
    """
    try:
        task_id = await vector_controller.reindex_vectors(
            document_ids=document_ids,
            model_name=model_name,
            user_id=current_user.id
        )
        return {"message": "Reindexing started", "task_id": task_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/status/{task_id}")
async def get_reindex_status(
    task_id: str,
    current_user = Depends(get_current_user)
):
    """
    Get status of a reindexing task.
    """
    status = await vector_controller.get_reindex_status(
        task_id=task_id,
        user_id=current_user.id
    )
    return status

@router.get("/metadata", response_model=VectorMetadata)
async def get_vector_metadata(
    current_user = Depends(get_current_user)
):
    """
    Get metadata about vector store (models, dimensions, etc.).
    """
    metadata = await vector_controller.get_vector_metadata(
        user_id=current_user.id
    )
    return metadata