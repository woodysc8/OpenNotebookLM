"""Tests for the asymmetric query/passage prefixes some embedding models need."""
from types import SimpleNamespace

import numpy as np
import pytest

from app.services import embeddings as embeddings_module
from app.services.cache import CacheService
from app.services.embeddings import EmbeddingService, prefix_for_role


@pytest.fixture
def model_name(monkeypatch):
    """Set the configured embedding model for one test."""
    def _set(name):
        monkeypatch.setattr(
            embeddings_module,
            "get_settings",
            lambda: SimpleNamespace(emb_model_name=name, emb_dimension=2),
        )
    return _set


def test_e5_models_get_asymmetric_prefixes(model_name):
    model_name("intfloat/multilingual-e5-base")

    assert prefix_for_role("query") == "query: "
    assert prefix_for_role("passage") == "passage: "
    # The two must differ, or the asymmetry the model was trained on is lost.
    assert prefix_for_role("query") != prefix_for_role("passage")


def test_non_e5_models_get_no_prefix(model_name):
    for name in ("BAAI/bge-m3", "BAAI/bge-small-en-v1.5",
                 "sentence-transformers/all-MiniLM-L6-v2"):
        model_name(name)
        assert prefix_for_role("query") == ""
        assert prefix_for_role("passage") == ""


def test_unknown_role_is_not_prefixed(model_name):
    model_name("intfloat/multilingual-e5-base")

    # Better to embed without a prefix than with an invented one.
    assert prefix_for_role("something-else") == ""


def test_cache_key_separates_model_role_and_normalization(
    model_name,
    monkeypatch,
) -> None:
    """Semantically different embedding requests cannot reuse one vector."""
    model = type(
        "Model",
        (),
        {
            "calls": 0,
            "encode": lambda self, _text, **_kwargs: (
                setattr(self, "calls", self.calls + 1)
                or np.array([float(self.calls), 1.0], dtype=np.float32)
            ),
        },
    )()
    service = object.__new__(EmbeddingService)
    cache = CacheService(
        redis_url=None,
        namespace="embedding-key-test",
        max_entries=100,
    )
    monkeypatch.setattr(EmbeddingService, "_model", model)
    monkeypatch.setattr(embeddings_module, "cache_service", cache)

    model_name("BAAI/bge-m3")
    service.generate_embedding(
        "same text",
        normalize=True,
        role="passage",
        document_id="document",
    )
    service.generate_embedding(
        "same text",
        normalize=True,
        role="other-role",
        document_id="document",
    )
    service.generate_embedding(
        "same text",
        normalize=False,
        role="passage",
        document_id="document",
    )
    model_name("another/model")
    service.generate_embedding(
        "same text",
        normalize=True,
        role="passage",
        document_id="document",
    )

    assert model.calls == 4


def test_openai_provider_batches_and_normalizes_embeddings(monkeypatch):
    """OpenAI embeddings use the configured model without local ML imports."""
    class FakeEmbeddings:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(data=[
                SimpleNamespace(index=1, embedding=[0.0, 3.0, 4.0]),
                SimpleNamespace(index=0, embedding=[3.0, 0.0, 4.0]),
            ])

    fake_embeddings = FakeEmbeddings()
    monkeypatch.setattr(
        embeddings_module,
        "get_settings",
        lambda: SimpleNamespace(
            embedding_provider="openai",
            emb_model_name="text-embedding-3-small",
            emb_dimension=3,
        ),
    )
    monkeypatch.setattr(EmbeddingService, "_client", SimpleNamespace(
        embeddings=fake_embeddings,
    ))
    monkeypatch.setattr(embeddings_module, "cache_service", None)

    service = object.__new__(EmbeddingService)
    result = service.generate_embedding(["first", "second"])

    assert fake_embeddings.calls == [{
        "model": "text-embedding-3-small",
        "input": ["first", "second"],
    }]
    np.testing.assert_allclose(result[0], [0.6, 0.0, 0.8])
    np.testing.assert_allclose(result[1], [0.0, 0.6, 0.8])
