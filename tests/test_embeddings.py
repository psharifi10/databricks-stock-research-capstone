"""Offline tests for embedding validation and lazy model loading."""

import math
import unittest
from unittest.mock import MagicMock

from pipelines.embeddings import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    EmbeddingError,
    QueryEmbeddingService,
    normalize_query_text,
    serialize_embedding,
    similarity_from_cosine_distance,
    validate_embedding,
)


def _vector(value: float = 0.0) -> list[float]:
    return [value] * EMBEDDING_DIMENSION


class EmbeddingHelperTests(unittest.TestCase):
    def test_embedding_requires_exactly_384_finite_numeric_values(self) -> None:
        self.assertEqual(len(validate_embedding(_vector())), 384)
        for invalid in (
            [0.0] * 383,
            [0.0] * 385,
            [0.0] * 383 + [math.inf],
            [0.0] * 383 + [True],
            "not-a-vector",
        ):
            with self.subTest(invalid_type=type(invalid).__name__):
                with self.assertRaises(EmbeddingError):
                    validate_embedding(invalid)

    def test_vector_serialization_is_validated_and_deterministic(self) -> None:
        serialized = serialize_embedding(_vector(0.25))

        self.assertTrue(serialized.startswith("["))
        self.assertTrue(serialized.endswith("]"))
        self.assertEqual(serialized.count(","), 383)
        self.assertNotIn("nan", serialized.lower())

    def test_query_normalization_rejects_blank_values(self) -> None:
        self.assertEqual(normalize_query_text("  Apple\n succession "), "Apple succession")
        for invalid in ("", " \t ", None):
            with self.assertRaises(EmbeddingError):
                normalize_query_text(invalid)

    def test_similarity_conversion_is_bounded(self) -> None:
        self.assertEqual(similarity_from_cosine_distance(0.25), 0.75)
        self.assertEqual(similarity_from_cosine_distance(-1.0), 1.0)
        self.assertEqual(similarity_from_cosine_distance(3.0), -1.0)

    def test_model_loading_is_lazy_and_injected(self) -> None:
        model = MagicMock()
        model.encode.return_value = [_vector(0.1)]
        factory = MagicMock(return_value=model)
        service = QueryEmbeddingService(factory)

        factory.assert_not_called()
        result = service.embed_query("  Apple leadership  ")

        factory.assert_called_once_with(EMBEDDING_MODEL_NAME)
        model.encode.assert_called_once_with(
            ["Apple leadership"],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.assertEqual(len(result), EMBEDDING_DIMENSION)

    def test_blank_query_does_not_load_model(self) -> None:
        factory = MagicMock()
        service = QueryEmbeddingService(factory)

        with self.assertRaises(EmbeddingError):
            service.embed_query("   ")

        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
