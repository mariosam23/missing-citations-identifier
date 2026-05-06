import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from database.qdrant import PaperVector, QdrantPaperStore, count_collections
from indexer import EmbeddingIndex


class FakeDenseModel:
    def __init__(self) -> None:
        self.encoded_texts: list[str] = []

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> list[list[float]]:
        self.encoded_texts.extend(texts)
        return [[float(i), float(i + 1)] for i, _text in enumerate(texts)]


class FakeSparseEmbedding:
    def __init__(self, offset: int) -> None:
        self.indices = [offset, offset + 1]
        self.values = [0.1, 0.2]


class FakeSparseModel:
    def embed(self, texts: list[str]) -> list[FakeSparseEmbedding]:
        return [FakeSparseEmbedding(i) for i, _text in enumerate(texts)]


class FakeStore(QdrantPaperStore):
    def __init__(self) -> None:
        self.collection = "papers"
        self.dense_dim = 1024
        self.collection_exists_calls = 0
        self.create_calls = 0
        self.count_calls = 0
        self.existing_calls: list[tuple[list[str], int]] = []
        self.upserted: list[PaperVector] = []
        self.existing_ids: set[str] = set()

    def create_collection_if_missing(self) -> bool:
        self.create_calls += 1
        return True

    def collection_exists(self) -> bool:
        self.collection_exists_calls += 1
        return True

    def count(self) -> int:
        self.count_calls += 1
        return 7

    def existing_point_ids(self, point_ids: list[str], batch_size: int = 64) -> set[str]:
        self.existing_calls.append((point_ids, batch_size))
        return self.existing_ids

    def upsert_paper_vectors(self, vectors: list[PaperVector]) -> int:
        self.upserted.extend(vectors)
        return len(vectors)


class EmbeddingIndexStoreTests(unittest.TestCase):
    def test_delegates_collection_methods_to_store(self) -> None:
        store = FakeStore()
        index = EmbeddingIndex(
            qdrant_client=None,
            dense_model=FakeDenseModel(),
            sparse_model=FakeSparseModel(),
            store=store,
        )

        self.assertTrue(index.create_collection_if_missing())
        self.assertTrue(index.collection_exists())
        self.assertEqual(index.count(), 7)
        self.assertEqual(store.create_calls, 1)
        self.assertEqual(store.collection_exists_calls, 1)
        self.assertEqual(store.count_calls, 1)

    def test_upsert_papers_filters_existing_and_delegates_vectors(self) -> None:
        store = FakeStore()
        dense = FakeDenseModel()
        index = EmbeddingIndex(
            qdrant_client=None,
            dense_model=dense,
            sparse_model=FakeSparseModel(),
            batch_size=2,
            store=store,
        )
        existing_id = index._stable_id("p1")
        store.existing_ids = {existing_id}

        count = index.upsert_papers(
            [
                {"paper_id": "p1", "title": "Old", "abstract": "Skip"},
                {
                    "paper_id": "p2",
                    "title": "New",
                    "abstract": "Index me",
                    "year": 2024,
                    "venue": "ACL",
                    "cited_by_count": 3,
                },
            ]
        )

        self.assertEqual(count, 1)
        self.assertEqual(dense.encoded_texts, ["passage: New. Index me"])
        self.assertEqual(len(store.upserted), 1)
        self.assertEqual(store.upserted[0].point_id, index._stable_id("p2"))
        self.assertEqual(store.upserted[0].payload["paper_id"], "p2")
        self.assertEqual(store.upserted[0].payload["year"], 2024)
        self.assertEqual(store.existing_calls[0][1], 2)


class FakeQdrantClient:
    def __init__(self) -> None:
        self.collections = [SimpleNamespace(name="papers")]
        self.created: dict | None = None
        self.upserted_collection: str | None = None
        self.upserted_points: list | None = None
        self.retrieved_ids: list[str] = []

    def get_collections(self):
        return SimpleNamespace(collections=self.collections)

    def create_collection(self, **kwargs):
        self.created = kwargs

    def count(self, collection_name: str, exact: bool):
        return SimpleNamespace(count=11)

    def retrieve(
        self,
        collection_name: str,
        ids: list[str],
        with_payload: bool,
        with_vectors: bool,
    ):
        self.retrieved_ids.extend(ids)
        return [SimpleNamespace(id=ids[0])] if ids else []

    def upsert(self, collection_name: str, points: list) -> None:
        self.upserted_collection = collection_name
        self.upserted_points = points


class QdrantPaperStoreTests(unittest.TestCase):
    def test_collection_helpers_delegate_to_client(self) -> None:
        client = FakeQdrantClient()
        store = QdrantPaperStore(client, collection="papers")

        self.assertEqual(count_collections(client), 1)
        self.assertTrue(store.collection_exists())
        self.assertFalse(store.create_collection_if_missing())
        self.assertEqual(store.count(), 11)

        client.collections = []
        self.assertTrue(store.create_collection_if_missing())
        created = client.created
        self.assertIsNotNone(created)
        assert created is not None
        self.assertEqual(created["collection_name"], "papers")

    def test_existing_point_ids_and_upsert_delegate_to_client(self) -> None:
        client = FakeQdrantClient()
        store = QdrantPaperStore(client, collection="papers")

        existing = store.existing_point_ids(["a", "b"], batch_size=1)
        self.assertEqual(existing, {"a", "b"})
        self.assertEqual(client.retrieved_ids, ["a", "b"])

        count = store.upsert_paper_vectors(
            [
                PaperVector(
                    point_id="p1",
                    dense=[0.1, 0.2],
                    sparse_indices=[1],
                    sparse_values=[0.5],
                    payload={"paper_id": "p1"},
                )
            ]
        )

        self.assertEqual(count, 1)
        self.assertEqual(client.upserted_collection, "papers")
        self.assertEqual(len(client.upserted_points or []), 1)


if __name__ == "__main__":
    unittest.main()
