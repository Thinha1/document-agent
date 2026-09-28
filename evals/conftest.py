import sys
from pathlib import Path

# evals/ sits next to tini/, not inside it — make both importable when
# running `pytest evals` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

# Retrieval tuning a maintainer keeps in .env (hybrid search, its weight, the
# rerank pool size, a non-default collection) must not decide what a test
# asserts about defaults: `Settings()` reads these at construction, so a tuned
# .env made tests of the defaults fail. Tests that need a value pass it in.
_RETRIEVAL_TUNING = (
    "TINI_DOCUMENT_HYBRID",
    "TINI_DOCUMENT_LEXICAL_WEIGHT",
    "TINI_DOCUMENT_FUSION",
    "TINI_RERANK_CANDIDATES",
    "TINI_QDRANT_COLLECTION",
)


@pytest.fixture(autouse=True)
def _hermetic_retrieval_settings(monkeypatch):
    for name in _RETRIEVAL_TUNING:
        monkeypatch.delenv(name, raising=False)
