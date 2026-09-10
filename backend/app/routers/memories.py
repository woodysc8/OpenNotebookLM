"""Authenticated CRUD and lexical search for personal memories."""
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import Memory, User
from app.routers.auth import get_memory_user
from app.schemas import (
    MemoryCreate,
    MemoryListResponse,
    MemoryResponse,
    MemorySearchRequest,
    MemoryUpdate,
)
from app.utils.time import utc_now

router = APIRouter(dependencies=[Depends(get_memory_user)])


def _response(memory: Memory) -> MemoryResponse:
    """Shape a database memory without exposing ORM-only names."""
    return MemoryResponse(
        id=memory.id,
        user_id=memory.user_id,
        category=memory.category,
        content=memory.content,
        source=memory.source,
        source_id=memory.source_id,
        importance=memory.importance,
        memory_key=memory.memory_key,
        metadata=memory.metadata_json or {},
        created_at=memory.created_at,
        updated_at=memory.updated_at,
    )


def _owned_memory(db: Session, memory_id: str, user: User) -> Memory:
    """Return a memory owned by the caller, hiding other users' rows."""
    memory = db.query(Memory).filter(
        Memory.id == memory_id,
        Memory.user_id == user.id,
    ).first()
    if memory is None:
        raise HTTPException(status_code=404, detail="Memory not found")
    return memory


@router.post("/memories", response_model=MemoryResponse)
async def create_memory(
    payload: MemoryCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_memory_user),
):
    """Create a memory or replace the caller's existing keyed memory."""
    memory = None
    if payload.memory_key:
        memory = db.query(Memory).filter(
            Memory.user_id == current_user.id,
            Memory.memory_key == payload.memory_key,
        ).first()
    if memory is None:
        memory = Memory(id=str(uuid4()), user_id=current_user.id)
        db.add(memory)
    memory.category = payload.category
    memory.content = payload.content
    memory.source = payload.source
    memory.source_id = payload.source_id
    memory.importance = payload.importance
    memory.memory_key = payload.memory_key
    memory.metadata_json = payload.metadata
    memory.updated_at = utc_now()
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Memory key already exists") from exc
    db.refresh(memory)
    return _response(memory)


@router.get("/memories", response_model=MemoryListResponse)
async def list_memories(
    category: str | None = Query(None, min_length=1, max_length=50),
    memory_key: str | None = Query(None, min_length=1, max_length=255),
    limit: int = Query(100, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_memory_user),
):
    """List the caller's memories with optional exact filters."""
    query = db.query(Memory).filter(Memory.user_id == current_user.id)
    if category:
        query = query.filter(Memory.category == category)
    if memory_key:
        query = query.filter(Memory.memory_key == memory_key)
    memories = query.order_by(Memory.importance.desc(), Memory.updated_at.desc()).limit(limit).all()
    return MemoryListResponse(memories=[_response(item) for item in memories], total=len(memories))


@router.get("/memories/{memory_id}", response_model=MemoryResponse)
async def get_memory(
    memory_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_memory_user),
):
    """Get one of the caller's memories."""
    return _response(_owned_memory(db, memory_id, current_user))


@router.patch("/memories/{memory_id}", response_model=MemoryResponse)
async def update_memory(
    memory_id: str,
    payload: MemoryUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_memory_user),
):
    """Update one of the caller's memories."""
    memory = _owned_memory(db, memory_id, current_user)
    changes = payload.model_dump(exclude_unset=True)
    if not changes:
        raise HTTPException(status_code=422, detail="At least one memory field is required")
    if changes.get("memory_key"):
        duplicate = db.query(Memory).filter(
            Memory.user_id == current_user.id,
            Memory.memory_key == changes["memory_key"],
            Memory.id != memory.id,
        ).first()
        if duplicate is not None:
            raise HTTPException(status_code=409, detail="Memory key already exists")
    if "metadata" in changes:
        memory.metadata_json = changes.pop("metadata")
    for field, value in changes.items():
        setattr(memory, field, value)
    memory.updated_at = utc_now()
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Memory key already exists") from exc
    db.refresh(memory)
    return _response(memory)


@router.delete("/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(
    memory_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_memory_user),
):
    """Forget one of the caller's memories."""
    memory = _owned_memory(db, memory_id, current_user)
    db.delete(memory)
    db.commit()


@router.post("/memories/search", response_model=MemoryListResponse)
async def search_memories(
    payload: MemorySearchRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_memory_user),
):
    """Search the caller's memories with simple lexical relevance."""
    query = db.query(Memory).filter(Memory.user_id == current_user.id)
    if payload.category:
        query = query.filter(Memory.category == payload.category)
    if payload.source:
        query = query.filter(Memory.source == payload.source)
    if payload.memory_key:
        query = query.filter(Memory.memory_key == payload.memory_key)
    words = [word.lower() for word in payload.query.split() if len(word) >= 3]
    if words:
        query = query.filter(or_(*(
            Memory.content.ilike("%%%s%%" % word) for word in words
        )))
    if payload.sort == "updated_at":
        ordering = (Memory.updated_at.desc(), Memory.id.desc())
    else:
        ordering = (Memory.importance.desc(), Memory.updated_at.desc(), Memory.id.desc())
    memories = query.order_by(*ordering).limit(payload.limit).all()
    return MemoryListResponse(memories=[_response(item) for item in memories], total=len(memories))