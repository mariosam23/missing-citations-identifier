import importlib
import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from database.postgres.base import Base  # noqa: E402
from database.postgres.tables.citation import Citation  # noqa: E402,F401
from database.postgres.tables.corpus_paper import CorpusPaper  # noqa: E402,F401
from database.postgres.tables.paper import Paper  # noqa: E402,F401
from entities import RankedPaper, RetrievalResult  # noqa: E402


class _BlockSentenceTransformersImport:
    def find_spec(self, fullname, path, target=None):
        if fullname == "sentence_transformers" or fullname.startswith(
            "sentence_transformers."
        ):
            raise AssertionError(
                "sentence_transformers should not be imported at module load time"
            )
        return None


class ModernizationTests(unittest.TestCase):
    def test_ranked_paper_serializes_contributions(self) -> None:
        result = RetrievalResult(
            paper_id="p1",
            title="Example title",
            score=0.5,
        )
        ranked = RankedPaper(
            result=result,
            aggregate_score=1.25,
            contributions={1: 2, 3: 4},
        )

        payload = ranked.to_dict()
        self.assertEqual(payload["aggregate_score"], 1.25)
        self.assertEqual(payload["contributions"], {1: 2, 3: 4})

    def test_sqlalchemy_models_register_with_declarative_base(self) -> None:
        self.assertIn("papers", Base.metadata.tables)
        self.assertIn("corpus_papers", Base.metadata.tables)
        self.assertIn("citations", Base.metadata.tables)

    def test_reranker_import_is_lazy(self) -> None:
        blocker = _BlockSentenceTransformersImport()
        sys.modules.pop("pipeline.reranker", None)
        sys.meta_path.insert(0, blocker)
        try:
            module = importlib.import_module("pipeline.reranker")
        finally:
            sys.meta_path.remove(blocker)

        reranker = module.CrossEncoderReranker(model=None)
        self.assertEqual(reranker.rerank("query", [], top_k=0), [])


if __name__ == "__main__":
    unittest.main()
