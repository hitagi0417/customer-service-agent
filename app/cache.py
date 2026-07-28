import hashlib
import json
import logging
from typing import Protocol

from redis import Redis
from redis.exceptions import RedisError

from app.config import settings
from app.schemas import KnowledgeMatch


logger = logging.getLogger(__name__)


class RetrievalCache(Protocol):
    """检索缓存的最小接口，便于替换实现和单元测试。"""

    def get(
        self,
        cache_key: str,
    ) -> list[KnowledgeMatch] | None:
        ...

    def set(
        self,
        cache_key: str,
        matches: list[KnowledgeMatch],
    ) -> None:
        ...

    def ping(self) -> bool:
        ...


class NullRetrievalCache:
    """未启用Redis时使用的空实现。"""

    def get(
        self,
        cache_key: str,
    ) -> list[KnowledgeMatch] | None:
        return None

    def set(
        self,
        cache_key: str,
        matches: list[KnowledgeMatch],
    ) -> None:
        return None

    def ping(self) -> bool:
        return True


class RedisRetrievalCache:
    """
    Redis检索缓存。

    Redis异常时采用fail-open：跳过缓存继续检索，
    避免非核心缓存故障直接拖垮客服主链路。
    """

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int,
        prefix: str = "customer-service:retrieval",
    ) -> None:
        self.client = Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
            health_check_interval=30,
        )
        self.ttl_seconds = ttl_seconds
        self.prefix = prefix

    def _redis_key(
        self,
        cache_key: str,
    ) -> str:
        digest = hashlib.sha256(
            cache_key.encode("utf-8")
        ).hexdigest()
        return f"{self.prefix}:{digest}"

    def get(
        self,
        cache_key: str,
    ) -> list[KnowledgeMatch] | None:
        try:
            value = self.client.get(
                self._redis_key(cache_key)
            )
        except RedisError as error:
            logger.warning(
                "Redis读取失败，已降级为实时检索：%s",
                type(error).__name__,
            )
            return None

        if value is None:
            return None

        try:
            data = json.loads(value)
            return [
                KnowledgeMatch.model_validate(item)
                for item in data
            ]
        except (ValueError, TypeError) as error:
            logger.warning(
                "Redis缓存内容无效，已忽略：%s",
                type(error).__name__,
            )
            return None

    def set(
        self,
        cache_key: str,
        matches: list[KnowledgeMatch],
    ) -> None:
        payload = json.dumps(
            [
                match.model_dump(mode="json")
                for match in matches
            ],
            ensure_ascii=False,
        )

        try:
            self.client.setex(
                self._redis_key(cache_key),
                self.ttl_seconds,
                payload,
            )
        except RedisError as error:
            logger.warning(
                "Redis写入失败，主链路继续执行：%s",
                type(error).__name__,
            )

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except RedisError:
            return False


def create_retrieval_cache() -> RetrievalCache:
    if not settings.enable_retrieval_cache:
        return NullRetrievalCache()

    if settings.redis_url is None:
        raise RuntimeError(
            "启用检索缓存时必须配置REDIS_URL"
        )

    return RedisRetrievalCache(
        redis_url=settings.redis_url,
        ttl_seconds=(
            settings.retrieval_cache_ttl_seconds
        ),
    )
