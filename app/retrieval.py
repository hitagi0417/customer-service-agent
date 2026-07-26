from typing import Literal

from sentence_transformers import (
    SentenceTransformer,
    util,
)

from app.bm25 import BM25Index
from app.config import settings
from app.knowledge import (
    KnowledgeChunk,
    load_knowledge_chunks,
)
from app.reranking import (
    CrossEncoderReranker,
    Reranker,
)
from app.schemas import KnowledgeMatch


RetrievalMode = Literal[
    "vector",
    "keyword",
    "hybrid",
    "hybrid_rerank",
]


class KnowledgeRetriever:
    """
    知识库BM25与向量混合检索器。

    它负责：
    1. 将知识片段转换成向量；
    2. 为知识片段建立BM25关键词索引；
    3. 分别召回语义候选和关键词候选；
    4. 融合两路分数并返回最相关片段。
    """

    def __init__(
        self,
        chunks: list[KnowledgeChunk],
        model_name: str,
        default_top_k: int = 3,
        default_min_score: float = 0.45,
        keyword_weight: float = 0.30,
        min_keyword_score: float = 0.35,
        candidate_multiplier: int = 4,
        reranker: Reranker | None = None,
        rerank_candidate_k: int = 10,
    ) -> None:
        if not chunks:
            raise ValueError("知识片段不能为空")

        if default_top_k <= 0:
            raise ValueError("default_top_k必须大于0")

        if not -1.0 <= default_min_score <= 1.0:
            raise ValueError(
                "default_min_score必须在-1到1之间"
            )
        if not 0.0 <= keyword_weight <= 1.0:
            raise ValueError(
                "keyword_weight必须在0到1之间"
            )
        if not 0.0 <= min_keyword_score <= 1.0:
            raise ValueError(
                "min_keyword_score必须在0到1之间"
            )
        if candidate_multiplier <= 0:
            raise ValueError(
                "candidate_multiplier必须大于0"
            )
        if rerank_candidate_k <= 0:
            raise ValueError(
                "rerank_candidate_k必须大于0"
            )

        self.chunks = chunks
        self.default_top_k = default_top_k
        self.default_min_score = default_min_score
        self.keyword_weight = keyword_weight
        self.min_keyword_score = min_keyword_score
        self.candidate_multiplier = candidate_multiplier
        self.reranker = reranker
        self.rerank_candidate_k = rerank_candidate_k

        print(f"正在加载向量模型：{model_name}")

        self.model = SentenceTransformer(model_name)

        self.corpus_embeddings = self._encode_chunks(
            self.chunks
        )
        self.keyword_index = self._build_keyword_index(
            self.chunks
        )

        print(
            "知识库混合索引构建完成，"
            f"共{len(self.chunks)}个片段"
        )

    def _encode_chunks(
        self,
        chunks: list[KnowledgeChunk],
    ):
        """
        将全部知识片段转换成向量。

        这个过程只在创建检索器或重新加载知识库时执行，
        不应该在每次用户提问时重复执行。
        """
        contents = [
            chunk.content
            for chunk in chunks
        ]

        return self.model.encode(
            contents,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _build_keyword_index(
        self,
        chunks: list[KnowledgeChunk],
    ) -> BM25Index:
        return BM25Index(
            [
                chunk.content
                for chunk in chunks
            ]
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        min_score: float | None = None,
        min_keyword_score: float | None = None,
        mode: RetrievalMode | None = None,
    ) -> list[KnowledgeMatch]:
        """
        搜索与用户问题最相关的知识片段。

        query：
            用户提出的问题。

        top_k：
            最多返回多少条结果。

        min_score：
            最低向量相似度。关键词分达到阈值时也可进入候选。

        mode：
            vector只使用向量，keyword只使用BM25，
            hybrid使用两路候选和融合分数，
            hybrid_rerank再使用Cross-Encoder精排。
        """
        cleaned_query = query.strip()

        if not cleaned_query:
            return []

        mode = mode or (
            "hybrid_rerank"
            if self.reranker is not None
            else "hybrid"
        )

        actual_top_k = (
            top_k
            if top_k is not None
            else self.default_top_k
        )

        actual_min_score = (
            min_score
            if min_score is not None
            else self.default_min_score
        )
        actual_min_keyword_score = (
            min_keyword_score
            if min_keyword_score is not None
            else self.min_keyword_score
        )

        if actual_top_k <= 0:
            raise ValueError("top_k必须大于0")

        if not -1.0 <= actual_min_score <= 1.0:
            raise ValueError(
                "min_score必须在-1到1之间"
            )
        if not 0.0 <= actual_min_keyword_score <= 1.0:
            raise ValueError(
                "min_keyword_score必须在0到1之间"
            )
        if mode not in {
            "vector",
            "keyword",
            "hybrid",
            "hybrid_rerank",
        }:
            raise ValueError(
                f"不支持的检索模式：{mode}"
            )
        if (
            mode == "hybrid_rerank"
            and self.reranker is None
        ):
            raise RuntimeError(
                "当前检索器没有配置Rerank模型"
            )

        # 先扩大候选集，再融合关键词和向量分数。
        actual_top_k = min(actual_top_k, len(self.chunks))
        candidate_k = min(
            len(self.chunks),
            max(
                actual_top_k,
                actual_top_k
                * self.candidate_multiplier,
            ),
        )

        vector_scores: dict[int, float] = {}

        if mode in {
            "vector",
            "hybrid",
            "hybrid_rerank",
        }:
            query_embedding = self.model.encode(
                [cleaned_query],
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            hits = util.semantic_search(
                query_embedding,
                self.corpus_embeddings,
                top_k=candidate_k,
            )[0]
            vector_scores = {
                int(hit["corpus_id"]): float(
                    hit["score"]
                )
                for hit in hits
            }

        keyword_scores = [
            0.0
            for _ in self.chunks
        ]

        if mode in {
            "keyword",
            "hybrid",
            "hybrid_rerank",
        }:
            keyword_scores = (
                self.keyword_index.normalized_scores(
                    cleaned_query
                )
            )

        keyword_candidate_ids = [
            corpus_id
            for corpus_id in sorted(
                range(len(keyword_scores)),
                key=lambda item: keyword_scores[item],
                reverse=True,
            )[:candidate_k]
            if keyword_scores[corpus_id] > 0.0
        ]

        if mode == "vector":
            candidate_ids = set(vector_scores)
        elif mode == "keyword":
            candidate_ids = set(
                keyword_candidate_ids
            )
        else:
            candidate_ids = (
                set(vector_scores)
                | set(keyword_candidate_ids)
            )

        matches: list[KnowledgeMatch] = []

        for corpus_id in candidate_ids:
            vector_score = vector_scores.get(corpus_id)
            keyword_score = keyword_scores[corpus_id]

            vector_passed = (
                vector_score is not None
                and vector_score >= actual_min_score
            )
            keyword_passed = (
                keyword_score
                >= actual_min_keyword_score
            )

            passed = (
                vector_passed
                if mode == "vector"
                else keyword_passed
                if mode == "keyword"
                else vector_passed or keyword_passed
            )

            if not passed:
                continue

            chunk = self.chunks[corpus_id]
            positive_vector_score = max(
                vector_score or 0.0,
                0.0,
            )
            final_score = (
                positive_vector_score
                if mode == "vector"
                else keyword_score
                if mode == "keyword"
                else (
                    (1.0 - self.keyword_weight)
                    * positive_vector_score
                    + self.keyword_weight
                    * keyword_score
                )
            )

            matches.append(
                KnowledgeMatch(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    content=chunk.content,
                    score=round(final_score, 4),
                    vector_score=(
                        round(vector_score, 4)
                        if vector_score is not None
                        else None
                    ),
                    keyword_score=round(
                        keyword_score,
                        4,
                    ),
                    retrieval_method=mode,
                    metadata=chunk.metadata,
                )
            )

        matches.sort(
            key=lambda match: (
                match.score,
                match.keyword_score,
                match.vector_score or -1.0,
            ),
            reverse=True,
        )

        if mode == "hybrid_rerank":
            rerank_candidates = matches[
                : self.rerank_candidate_k
            ]
            rerank_scores = self.reranker.score(
                query=cleaned_query,
                documents=[
                    match.content
                    for match in rerank_candidates
                ],
            )

            for match, rerank_score in zip(
                rerank_candidates,
                rerank_scores,
                strict=True,
            ):
                match.rerank_score = round(
                    rerank_score,
                    4,
                )
                match.retrieval_method = (
                    "hybrid_rerank"
                )

            rerank_candidates.sort(
                key=lambda match: (
                    match.rerank_score
                    if match.rerank_score is not None
                    else float("-inf")
                ),
                reverse=True,
            )
            matches = rerank_candidates

        return matches[:actual_top_k]

    def reload(
        self,
        chunks: list[KnowledgeChunk],
    ) -> None:
        """
        重新构建知识库索引。

        后面知识文件发生变化时，
        不需要重启整个程序即可更新索引。
        """
        if not chunks:
            raise ValueError("新的知识片段不能为空")

        self.chunks = chunks
        self.corpus_embeddings = self._encode_chunks(
            self.chunks
        )
        self.keyword_index = self._build_keyword_index(
            self.chunks
        )

        print(
            "知识库混合索引重新加载完成，"
            f"共{len(self.chunks)}个片段"
        )


def create_knowledge_retriever(
    enable_reranker: bool | None = None,
) -> KnowledgeRetriever:
    """
    使用项目统一配置创建知识库检索器。
    """
    chunks = load_knowledge_chunks()

    should_enable_reranker = (
        settings.rag_enable_reranker
        if enable_reranker is None
        else enable_reranker
    )
    reranker = (
        CrossEncoderReranker(
            settings.reranker_model_name
        )
        if should_enable_reranker
        else None
    )

    return KnowledgeRetriever(
        chunks=chunks,
        model_name=settings.embedding_model_name,
        default_top_k=settings.rag_top_k,
        default_min_score=settings.rag_min_score,
        keyword_weight=settings.rag_keyword_weight,
        min_keyword_score=(
            settings.rag_min_keyword_score
        ),
        candidate_multiplier=(
            settings.rag_candidate_multiplier
        ),
        reranker=reranker,
        rerank_candidate_k=(
            settings.rerank_candidate_k
        ),
    )
