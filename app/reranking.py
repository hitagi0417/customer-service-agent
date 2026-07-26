from typing import Protocol

from sentence_transformers import CrossEncoder


class Reranker(Protocol):
    def score(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        ...


class CrossEncoderReranker:
    """使用问题和候选片段联合编码的二阶段精排器。"""

    def __init__(
        self,
        model_name: str,
        max_length: int = 512,
    ) -> None:
        print(f"正在加载Rerank模型：{model_name}")
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
        )

    def score(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        if not documents:
            return []

        scores = self.model.predict(
            [
                [query, document]
                for document in documents
            ],
            show_progress_bar=False,
        )

        return [
            float(score)
            for score in scores
        ]
