"""Authenticated personal-memory API contracts."""
import pytest
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import IntegrityError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import get_db
from app.config import Settings, get_settings
from app.db.models import Base, Memory, User
from app.routers import memories
from conftest import authenticated_client


app = FastAPI()
app.include_router(memories.router, prefix="/api")
engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


@pytest.fixture
def clients():
    """Return two authenticated clients sharing an isolated database."""
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    app.dependency_overrides[get_db] = lambda: session
    alice = authenticated_client(app, session, "memory-alice")
    bob = authenticated_client(app, session, "memory-bob")
    alice_user = session.query(User).filter_by(username="memory-alice").one()
    settings = Settings(
        second_brain_service_token="service-secret",
        second_brain_service_user_id=alice_user.id,
    )
    app.dependency_overrides[get_settings] = lambda: settings
    service = TestClient(app, headers={
        "Authorization": "Bearer service-secret",
        "X-Second-Brain-User": alice_user.id,
    })
    yield alice, bob, service, alice_user.id
    app.dependency_overrides.clear()
    session.close()


def memory_payload(content="Sam prefers Hyatt hotels.", key="hotel-preference"):
    """Build a representative memory payload."""
    return {
        "category": "preference",
        "content": content,
        "source": "user",
        "importance": 5,
        "memory_key": key,
        "metadata": {"explicit": True},
    }


def test_memory_crud_and_keyed_replacement(clients):
    alice, _bob, _service, _alice_id = clients
    created = alice.post("/api/memories", json=memory_payload()).json()
    assert created["content"] == "Sam prefers Hyatt hotels."
    assert created["metadata"] == {"explicit": True}

    replaced = alice.post(
        "/api/memories",
        json=memory_payload("Sam prefers Marriott hotels."),
    ).json()
    assert replaced["id"] == created["id"]
    assert alice.get("/api/memories").json()["total"] == 1

    updated = alice.patch(
        f"/api/memories/{created['id']}",
        json={"importance": 9, "category": "travel"},
    )
    assert updated.status_code == 200
    assert updated.json()["category"] == "travel"

    deleted = alice.delete(f"/api/memories/{created['id']}")
    assert deleted.status_code == 204
    assert alice.get(f"/api/memories/{created['id']}").status_code == 404


def test_search_category_and_cross_user_isolation(clients):
    alice, bob, _service, _alice_id = clients
    created = alice.post("/api/memories", json=memory_payload()).json()
    alice.post("/api/memories", json={**memory_payload("Sam works at StreetCred.", "work"), "category": "work"})

    result = alice.post("/api/memories/search", json={"query": "hotel", "category": "preference"})
    assert [item["id"] for item in result.json()["memories"]] == [created["id"]]
    assert bob.get(f"/api/memories/{created['id']}").status_code == 404
    assert bob.patch(f"/api/memories/{created['id']}", json={"content": "stolen"}).status_code == 404
    assert bob.delete(f"/api/memories/{created['id']}").status_code == 404


def test_authentication_and_invalid_input(clients):
    alice, _bob, _service, _alice_id = clients
    assert alice.post("/api/memories", json={"category": "preference"}).status_code == 422
    assert TestClient(app).get("/api/memories").status_code == 401


def test_service_token_authentication_and_fixed_user_mapping(clients):
    _alice, bob, service, alice_id = clients
    created = service.post("/api/memories", json=memory_payload()).json()
    assert created["user_id"] == alice_id
    assert service.get(f"/api/memories/{created['id']}").status_code == 200

    assert TestClient(app, headers={
        "Authorization": "Bearer invalid-service-token",
        "X-Second-Brain-User": alice_id,
    }).get("/api/memories").status_code == 401
    assert TestClient(app, headers={
        "Authorization": "Bearer service-secret",
        "X-Second-Brain-User": "not-alice",
    }).get("/api/memories").status_code == 403
    assert bob.get(f"/api/memories/{created['id']}").status_code == 404


def test_normal_jwt_authentication_still_works(clients):
    alice, _bob, _service, _alice_id = clients
    response = alice.post("/api/memories", json=memory_payload())
    assert response.status_code == 200


def test_same_user_key_is_unique_and_other_users_may_reuse_it(clients):
    alice, bob, _service, _alice_id = clients
    first = alice.post("/api/memories", json=memory_payload()).json()
    bob_memory = bob.post("/api/memories", json=memory_payload()).json()
    assert bob_memory["id"] != first["id"]

    with TestingSessionLocal() as session:
        user = session.query(User).filter_by(username="memory-alice").one()
        session.add(Memory(
            id="duplicate-key",
            user_id=user.id,
            category="preference",
            content="duplicate",
            source="test",
            memory_key="hotel-preference",
        ))
        with pytest.raises(IntegrityError):
            session.commit()


def test_updated_at_sort_beats_importance_for_latest_selection(clients):
    _alice, _bob, service, _alice_id = clients
    older = service.post("/api/memories", json=memory_payload("older", "older-key")).json()
    newer = service.post("/api/memories", json=memory_payload("newer", "newer-key")).json()
    with TestingSessionLocal() as session:
        session.query(Memory).filter_by(id=older["id"]).update({
            "importance": 100,
            "updated_at": datetime(2026, 1, 1, tzinfo=timezone.utc),
        })
        session.query(Memory).filter_by(id=newer["id"]).update({
            "importance": 0,
            "updated_at": datetime(2026, 2, 1, tzinfo=timezone.utc),
        })
        session.commit()

    result = service.post("/api/memories/search", json={"sort": "updated_at", "limit": 1})
    assert result.json()["memories"][0]["id"] == newer["id"]