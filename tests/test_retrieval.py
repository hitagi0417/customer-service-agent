from pathlib import Path

import pytest

from app.knowledge import (
    KnowledgeChunk,
    KnowledgeLoader,
)
from app.retrieval import KnowledgeRetriever


class FakeEmbeddingModel:
    """
    模拟向量模型，避免测试时下载和加载真实模型。
    """

    def encode(
        self,
        texts,
        **kwargs,
    ):
        return {
            "texts": texts,
        }


class FakeReranker:
    def score(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        return [
            1.0
            if "标准版" in document
            else 0.1
            for document in documents
        ]


def test_knowledge_loader_reads_txt(
    tmp_path: Path,
) -> None:
    knowledge_file = (
        tmp_path / "company.txt"
    )

    knowledge_file.write_text(
        (
            "公司成立于2024年。\n"
            "客服工作时间是周一到周五。\n"
            "退款需要在购买后7天内申请。"
        ),
        encoding="utf-8",
    )

    loader = KnowledgeLoader(
        knowledge_dir=tmp_path,
        chunk_size=30,
        chunk_overlap=5,
    )

    chunks = loader.load()

    assert len(chunks) >= 1

    assert all(
        chunk.source == "company.txt"
        for chunk in chunks
    )

    assert all(
        chunk.content
        for chunk in chunks
    )


def test_chunk_ids_are_stable(
    tmp_path: Path,
) -> None:
    knowledge_file = (
        tmp_path / "policy.txt"
    )

    knowledge_file.write_text(
        "退款申请需要在购买后7天内提交。",
        encoding="utf-8",
    )

    loader = KnowledgeLoader(
        knowledge_dir=tmp_path
    )

    first_load = loader.load()
    second_load = loader.load()

    first_ids = [
        chunk.chunk_id
        for chunk in first_load
    ]

    second_ids = [
        chunk.chunk_id
        for chunk in second_load
    ]

    assert first_ids == second_ids


def test_invalid_chunk_configuration(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="chunk_overlap必须小于chunk_size",
    ):
        KnowledgeLoader(
            knowledge_dir=tmp_path,
            chunk_size=100,
            chunk_overlap=100,
        )


def test_retriever_filters_low_scores(
    monkeypatch,
) -> None:
    chunks = [
        KnowledgeChunk(
            chunk_id="chunk_1",
            source="company.txt",
            content="客服工作时间是周一到周五。",
        ),
        KnowledgeChunk(
            chunk_id="chunk_2",
            source="refund.txt",
            content="退款申请需要在7天内提交。",
        ),
    ]

    fake_model = FakeEmbeddingModel()

    monkeypatch.setattr(
        "app.retrieval.SentenceTransformer",
        lambda model_name: fake_model,
    )

    def fake_semantic_search(
        query_embedding,
        corpus_embeddings,
        top_k,
    ):
        return [
            [
                {
                    "corpus_id": 0,
                    "score": 0.85,
                },
                {
                    "corpus_id": 1,
                    "score": 0.30,
                },
            ][:top_k]
        ]

    monkeypatch.setattr(
        "app.retrieval.util.semantic_search",
        fake_semantic_search,
    )

    retriever = KnowledgeRetriever(
        chunks=chunks,
        model_name="fake-model",
        default_top_k=2,
        default_min_score=0.45,
    )

    matches = retriever.search(
        "客服几点上班？"
    )

    assert len(matches) == 1
    assert matches[0].chunk_id == "chunk_1"
    assert matches[0].score == 0.895
    assert matches[0].vector_score == 0.85
    assert matches[0].keyword_score == 1.0


def test_retriever_can_override_threshold(
    monkeypatch,
) -> None:
    chunks = [
        KnowledgeChunk(
            chunk_id="chunk_1",
            source="company.txt",
            content="测试知识",
        )
    ]

    fake_model = FakeEmbeddingModel()

    monkeypatch.setattr(
        "app.retrieval.SentenceTransformer",
        lambda model_name: fake_model,
    )

    monkeypatch.setattr(
        "app.retrieval.util.semantic_search",
        lambda *args, **kwargs: [
            [
                {
                    "corpus_id": 0,
                    "score": 0.2,
                }
            ]
        ],
    )

    retriever = KnowledgeRetriever(
        chunks=chunks,
        model_name="fake-model",
        default_min_score=0.45,
    )

    default_matches = retriever.search(
        "完全无关内容"
    )

    all_matches = retriever.search(
        "完全无关内容",
        min_score=-1.0,
    )

    assert default_matches == []
    assert len(all_matches) == 1


def test_keyword_match_can_rescue_low_vector_score(
    monkeypatch,
) -> None:
    chunks = [
        KnowledgeChunk(
            chunk_id="product_xj_900",
            source="products.md",
            content="XJ-900型号产品支持两年保修。",
        ),
        KnowledgeChunk(
            chunk_id="refund",
            source="refund.md",
            content="普通商品支持七天退款。",
        ),
    ]
    fake_model = FakeEmbeddingModel()

    monkeypatch.setattr(
        "app.retrieval.SentenceTransformer",
        lambda model_name: fake_model,
    )
    monkeypatch.setattr(
        "app.retrieval.util.semantic_search",
        lambda *args, **kwargs: [
            [
                {
                    "corpus_id": 0,
                    "score": 0.20,
                },
                {
                    "corpus_id": 1,
                    "score": 0.10,
                },
            ]
        ],
    )

    retriever = KnowledgeRetriever(
        chunks=chunks,
        model_name="fake-model",
        default_min_score=0.45,
        min_keyword_score=0.35,
    )
    matches = retriever.search(
        "XJ-900保修多久？"
    )

    assert len(matches) == 1
    assert matches[0].chunk_id == "product_xj_900"
    assert matches[0].vector_score == 0.2
    assert matches[0].keyword_score == 1.0

    keyword_matches = retriever.search(
        "XJ-900保修多久？",
        mode="keyword",
    )

    assert keyword_matches[0].retrieval_method == "keyword"
    assert keyword_matches[0].score == 1.0

    with pytest.raises(
        ValueError,
        match="不支持的检索模式",
    ):
        retriever.search(
            "测试",
            mode="rerank",  # type: ignore[arg-type]
        )


def test_reranker_reorders_hybrid_candidates(
    monkeypatch,
) -> None:
    chunks = [
        KnowledgeChunk(
            chunk_id="pro",
            source="pro.md",
            content="XJ-900 Pro保修3年。",
        ),
        KnowledgeChunk(
            chunk_id="standard",
            source="standard.md",
            content="标准版XJ-900保修2年。",
        ),
    ]
    fake_model = FakeEmbeddingModel()

    monkeypatch.setattr(
        "app.retrieval.SentenceTransformer",
        lambda model_name: fake_model,
    )
    monkeypatch.setattr(
        "app.retrieval.util.semantic_search",
        lambda *args, **kwargs: [
            [
                {"corpus_id": 0, "score": 0.9},
                {"corpus_id": 1, "score": 0.8},
            ]
        ],
    )
    retriever = KnowledgeRetriever(
        chunks=chunks,
        model_name="fake-model",
        default_min_score=-1.0,
        min_keyword_score=0.0,
        reranker=FakeReranker(),
    )

    matches = retriever.search(
        "标准版XJ-900保修多久？",
        top_k=2,
    )

    assert matches[0].source == "standard.md"
    assert matches[0].rerank_score == 1.0
    assert (
        matches[0].retrieval_method
        == "hybrid_rerank"
    )
