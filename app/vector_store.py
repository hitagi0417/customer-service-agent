import hashlib
import json
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

from qdrant_client import QdrantClient, models

from app.knowledge import KnowledgeChunk


@dataclass(frozen=True)
class VectorHit:
    chunk_id: str
    score: float


@dataclass(frozen=True)
class VectorSyncResult:
    total_chunks: int
    upserted_chunks: int
    deleted_chunks: int


class VectorStore(Protocol):
    def sync(
        self,
        chunks: list[KnowledgeChunk],
        embedding_model,
        embedding_model_name: str,
    ) -> VectorSyncResult:
        ...

    def search(
        self,
        query_vector: list[float],
        top_k: int,
    ) -> list[VectorHit]:
        ...

    def ping(self) -> bool:
        ...


class QdrantVectorStore:
    """
    Qdrant向量存储。

    chunk_id由知识内容稳定生成；同步时只计算新增或变更片段，
    删除知识后也会删除Qdrant中的孤儿向量。
    """

    def __init__(
        self,
        url: str,
        collection_name: str,
        api_key: str | None = None,
        timeout_seconds: float = 10,
        client: QdrantClient | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.client = client or QdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout_seconds,
        )

    def sync(
        self,
        chunks: list[KnowledgeChunk],
        embedding_model,
        embedding_model_name: str,
    ) -> VectorSyncResult:
        if not chunks:
            raise ValueError("Qdrant同步的知识片段不能为空")

        collection_exists = self.client.collection_exists(
            self.collection_name
        )
        existing = (
            self._load_existing_payloads()
            if collection_exists
            else {}
        )
        current = {
            chunk.chunk_id: self._content_hash(chunk)
            for chunk in chunks
        }
        to_upsert = [
            chunk
            for chunk in chunks
            if (
                chunk.chunk_id not in existing
                or existing[chunk.chunk_id].get(
                    "content_hash"
                )
                != current[chunk.chunk_id]
                or existing[chunk.chunk_id].get(
                    "embedding_model"
                )
                != embedding_model_name
            )
        ]
        stale_chunk_ids = (
            set(existing) - set(current)
        )

        embeddings: list[list[float]] = []
        if to_upsert:
            encoded = embedding_model.encode(
                [
                    chunk.content
                    for chunk in to_upsert
                ],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            embeddings = [
                self._to_vector(item)
                for item in encoded
            ]

        if not collection_exists:
            if not embeddings:
                raise RuntimeError(
                    "创建Qdrant集合时未生成任何向量"
                )
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=len(embeddings[0]),
                    distance=models.Distance.COSINE,
                ),
            )
        elif embeddings:
            expected_size = self._vector_size()
            actual_size = len(embeddings[0])

            if expected_size != actual_size:
                raise RuntimeError(
                    "Embedding维度与现有Qdrant集合不一致："
                    f"集合={expected_size}，模型={actual_size}。"
                    "请更换QDRANT_COLLECTION后重新入库。"
                )

        for start in range(0, len(to_upsert), 128):
            chunk_batch = to_upsert[start : start + 128]
            vector_batch = embeddings[start : start + 128]
            points = [
                models.PointStruct(
                    id=self._point_id(chunk.chunk_id),
                    vector=vector,
                    payload={
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "content_hash": (
                            current[chunk.chunk_id]
                        ),
                        "embedding_model": (
                            embedding_model_name
                        ),
                        "metadata": (
                            self._json_safe(chunk.metadata)
                        ),
                    },
                )
                for chunk, vector in zip(
                    chunk_batch,
                    vector_batch,
                    strict=True,
                )
            ]
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

        if stale_chunk_ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=[
                        self._point_id(chunk_id)
                        for chunk_id in stale_chunk_ids
                    ]
                ),
                wait=True,
            )

        return VectorSyncResult(
            total_chunks=len(chunks),
            upserted_chunks=len(to_upsert),
            deleted_chunks=len(stale_chunk_ids),
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int,
    ) -> list[VectorHit]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        hits: list[VectorHit] = []

        for point in response.points:
            payload = point.payload or {}
            chunk_id = payload.get("chunk_id")

            if isinstance(chunk_id, str):
                hits.append(
                    VectorHit(
                        chunk_id=chunk_id,
                        score=float(point.score),
                    )
                )

        return hits

    def ping(self) -> bool:
        try:
            self.client.get_collection(
                self.collection_name
            )
            return True
        except Exception:
            return False

    def _load_existing_payloads(
        self,
    ) -> dict[str, dict[str, Any]]:
        existing: dict[str, dict[str, Any]] = {}
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = dict(point.payload or {})
                chunk_id = payload.get("chunk_id")

                if isinstance(chunk_id, str):
                    existing[chunk_id] = payload

            if offset is None:
                break

        return existing

    def _vector_size(self) -> int:
        collection = self.client.get_collection(
            self.collection_name
        )
        vectors = collection.config.params.vectors

        if isinstance(vectors, dict):
            raise RuntimeError(
                "当前项目不支持Qdrant命名向量集合"
            )

        return int(vectors.size)

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"customer-service:{chunk_id}",
            )
        )

    @staticmethod
    def _content_hash(
        chunk: KnowledgeChunk,
    ) -> str:
        return hashlib.sha256(
            chunk.content.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _json_safe(
        value: dict[str, Any],
    ) -> dict[str, Any]:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                default=str,
            )
        )

    @staticmethod
    def _to_vector(value) -> list[float]:
        if hasattr(value, "tolist"):
            value = value.tolist()
        return [float(item) for item in value]
