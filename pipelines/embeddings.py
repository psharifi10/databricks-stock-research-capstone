"""Spark-independent embedding constants, validation, and lazy model loading."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import math
from numbers import Real
from typing import Any


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
ModelFactory = Callable[[str], Any]


class EmbeddingError(ValueError):
    """Raised when embedding input or output is unusable."""


def normalize_query_text(value: Any) -> str:
    """Normalize a semantic query and reject blank or non-string values."""

    if not isinstance(value, str):
        raise EmbeddingError("Semantic search query must be a string.")
    normalized = " ".join(value.split())
    if not normalized:
        raise EmbeddingError("Semantic search query cannot be blank.")
    return normalized


def validate_embedding(vector: Iterable[Any]) -> tuple[float, ...]:
    """Return a finite 384-value vector or raise a client-safe error."""

    if isinstance(vector, (str, bytes)):
        raise EmbeddingError("Embedding must be a numeric vector.")
    try:
        values = list(vector)
    except TypeError as error:
        raise EmbeddingError("Embedding must be a numeric vector.") from error
    if len(values) != EMBEDDING_DIMENSION:
        raise EmbeddingError(
            f"Embedding must contain exactly {EMBEDDING_DIMENSION} values."
        )

    normalized: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise EmbeddingError("Embedding values must be finite numbers.")
        number = float(value)
        if not math.isfinite(number):
            raise EmbeddingError("Embedding values must be finite numbers.")
        normalized.append(number)
    return tuple(normalized)


def serialize_embedding(vector: Iterable[Any]) -> str:
    """Serialize a validated vector for a parameterized PostgreSQL cast."""

    values = validate_embedding(vector)
    return "[" + ",".join(format(value, ".17g") for value in values) + "]"


def similarity_from_cosine_distance(distance: Any) -> float:
    """Convert cosine distance to a bounded similarity value."""

    if isinstance(distance, bool) or not isinstance(distance, Real):
        raise EmbeddingError("Cosine distance must be a finite number.")
    numeric_distance = float(distance)
    if not math.isfinite(numeric_distance):
        raise EmbeddingError("Cosine distance must be a finite number.")
    return max(-1.0, min(1.0, 1.0 - numeric_distance))


class QueryEmbeddingService:
    """Lazily load one sentence-transformer model for query embeddings."""

    def __init__(self, model_factory: ModelFactory | None = None) -> None:
        self._model_factory = model_factory or _load_sentence_transformer
        self._model: Any | None = None

    def embed_query(self, query: Any) -> tuple[float, ...]:
        normalized_query = normalize_query_text(query)
        encoded = self._get_model().encode(
            [normalized_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        try:
            vectors = list(encoded)
        except TypeError as error:
            raise EmbeddingError("Embedding model returned an unusable value.") from error
        if len(vectors) != 1:
            raise EmbeddingError("Embedding model returned an unusable value.")
        return validate_embedding(vectors[0])

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = self._model_factory(EMBEDDING_MODEL_NAME)
        return self._model


def _load_sentence_transformer(model_name: str) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)
