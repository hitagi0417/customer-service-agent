import math
import re
from collections import Counter


TOKEN_PATTERN = re.compile(
    r"[a-zA-Z0-9_./-]+|[\u4e00-\u9fff]+"
)


def tokenize_for_search(text: str) -> list[str]:
    """
    将中英文客服文本转换成适合关键词检索的词项。

    英文、数字和产品编号保留为完整词项；连续中文使用二元组，
    不依赖额外分词模型也能匹配“退款时间”“客服电话”等短语。
    """
    tokens: list[str] = []

    for segment in TOKEN_PATTERN.findall(
        text.lower()
    ):
        if "\u4e00" <= segment[0] <= "\u9fff":
            if len(segment) == 1:
                tokens.append(segment)
            else:
                tokens.extend(
                    segment[index : index + 2]
                    for index in range(
                        len(segment) - 1
                    )
                )
        else:
            tokens.append(segment)

    return tokens


class BM25Index:
    """轻量BM25关键词索引，用于和向量召回组成混合检索。"""

    def __init__(
        self,
        documents: list[str],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not documents:
            raise ValueError("BM25文档不能为空")
        if k1 <= 0:
            raise ValueError("BM25参数k1必须大于0")
        if not 0.0 <= b <= 1.0:
            raise ValueError("BM25参数b必须在0到1之间")

        self.k1 = k1
        self.b = b
        self.document_tokens = [
            tokenize_for_search(document)
            for document in documents
        ]
        self.term_frequencies = [
            Counter(tokens)
            for tokens in self.document_tokens
        ]
        self.document_lengths = [
            len(tokens)
            for tokens in self.document_tokens
        ]
        self.average_document_length = (
            sum(self.document_lengths)
            / len(self.document_lengths)
        )

        document_frequencies: Counter[str] = Counter()

        for tokens in self.document_tokens:
            document_frequencies.update(set(tokens))

        document_count = len(documents)
        self.inverse_document_frequencies = {
            term: math.log(
                1.0
                + (
                    document_count
                    - frequency
                    + 0.5
                )
                / (frequency + 0.5)
            )
            for term, frequency
            in document_frequencies.items()
        }

    def score(self, query: str) -> list[float]:
        query_tokens = tokenize_for_search(query)

        if not query_tokens:
            return [
                0.0
                for _ in self.document_tokens
            ]

        scores: list[float] = []

        for term_frequency, document_length in zip(
            self.term_frequencies,
            self.document_lengths,
            strict=True,
        ):
            score = 0.0

            for term in query_tokens:
                frequency = term_frequency.get(term, 0)

                if frequency == 0:
                    continue

                inverse_frequency = (
                    self.inverse_document_frequencies.get(
                        term,
                        0.0,
                    )
                )
                length_ratio = (
                    document_length
                    / self.average_document_length
                    if self.average_document_length > 0
                    else 0.0
                )
                denominator = (
                    frequency
                    + self.k1
                    * (
                        1.0
                        - self.b
                        + self.b * length_ratio
                    )
                )
                score += (
                    inverse_frequency
                    * frequency
                    * (self.k1 + 1.0)
                    / denominator
                )

            scores.append(score)

        return scores

    def normalized_scores(
        self,
        query: str,
    ) -> list[float]:
        """将当前查询的BM25分数缩放到0到1，便于融合。"""
        scores = self.score(query)
        maximum = max(scores, default=0.0)

        if maximum <= 0.0:
            return [
                0.0
                for _ in scores
            ]

        return [
            score / maximum
            for score in scores
        ]
