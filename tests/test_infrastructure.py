from qdrant_client import QdrantClient

from app.knowledge import KnowledgeChunk
from app.retrieval import KnowledgeRetriever
from app.schemas import KnowledgeMatch
from app.vector_store import QdrantVectorStore


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.encoded_text_count = 0

    def encode(
        self,
        texts,
        **kwargs,
    ):
        self.encoded_text_count += len(texts)
        vectors = []

        for text in texts:
            if "退款" in text:
                vectors.append([0.0, 1.0, 0.0])
            elif "更新" in text:
                vectors.append([0.0, 0.0, 1.0])
            else:
                vectors.append([1.0, 0.0, 0.0])

        return vectors


class FakeRetrievalCache:
    def __init__(self) -> None:
        self.values: dict[
            str,
            list[KnowledgeMatch],
        ] = {}

    def get(
        self,
        cache_key: str,
    ) -> list[KnowledgeMatch] | None:
        matches = self.values.get(cache_key)

        if matches is None:
            return None

        return [
            match.model_copy(deep=True)
            for match in matches
        ]

    def set(
        self,
        cache_key: str,
        matches: list[KnowledgeMatch],
    ) -> None:
        self.values[cache_key] = [
            match.model_copy(deep=True)
            for match in matches
        ]

    def ping(self) -> bool:
        return True


def test_qdrant_incremental_sync_and_delete() -> None:
    client = QdrantClient(":memory:")
    store = QdrantVectorStore(
        url="http://unused",
        collection_name="test_knowledge",
        client=client,
    )
    model = FakeEmbeddingModel()
    initial_chunks = [
        KnowledgeChunk(
            chunk_id="company",
            source="company.md",
            content="客服工作时间是工作日。",
        ),
        KnowledgeChunk(
            chunk_id="refund",
            source="refund.md",
            content="退款需要在七天内申请。",
        ),
    ]

    first = store.sync(
        chunks=initial_chunks,
        embedding_model=model,
        embedding_model_name="fake-v1",
    )
    second = store.sync(
        chunks=initial_chunks,
        embedding_model=model,
        embedding_model_name="fake-v1",
    )

    assert first.upserted_chunks == 2
    assert second.upserted_chunks == 0
    assert model.encoded_text_count == 2

    changed_chunks = [
        KnowledgeChunk(
            chunk_id="company",
            source="company.md",
            content="客服工作时间已更新。",
        )
    ]
    third = store.sync(
        chunks=changed_chunks,
        embedding_model=model,
        embedding_model_name="fake-v1",
    )

    assert third.upserted_chunks == 1
    assert third.deleted_chunks == 1
    assert model.encoded_text_count == 3

    hits = store.search(
        query_vector=[0.0, 0.0, 1.0],
        top_k=1,
    )

    assert len(hits) == 1
    assert hits[0].chunk_id == "company"


def test_retrieval_cache_uses_knowledge_fingerprint(
    monkeypatch,
) -> None:
    semantic_search_calls = 0

    class MemoryEmbeddingModel:
        def encode(self, texts, **kwargs):
            return {"texts": texts}

    def fake_semantic_search(
        query_embedding,
        corpus_embeddings,
        top_k,
    ):
        nonlocal semantic_search_calls
        semantic_search_calls += 1
        return [[{"corpus_id": 0, "score": 0.9}]]

    monkeypatch.setattr(
        "app.retrieval.SentenceTransformer",
        lambda model_name: MemoryEmbeddingModel(),
    )
    monkeypatch.setattr(
        "app.retrieval.util.semantic_search",
        fake_semantic_search,
    )
    cache = FakeRetrievalCache()
    retriever = KnowledgeRetriever(
        chunks=[
            KnowledgeChunk(
                chunk_id="company-v1",
                source="company.md",
                content="客服工作时间是工作日。",
            )
        ],
        model_name="fake-model",
        cache=cache,
    )

    first = retriever.search("客服工作时间")
    second = retriever.search("客服工作时间")

    assert first == second
    assert semantic_search_calls == 1

    retriever.reload(
        [
            KnowledgeChunk(
                chunk_id="company-v2",
                source="company.md",
                content="客服工作时间更新为每天。",
            )
        ]
    )
    retriever.search("客服工作时间")

    assert semantic_search_calls == 2
