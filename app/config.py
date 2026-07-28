import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


# config.py 位于：项目根目录/app/config.py
# parent 是 app，parent.parent 才是项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 加载项目根目录中的 .env
load_dotenv(PROJECT_ROOT / ".env")


def get_required_env(name: str) -> str:
    """
    获取必填的环境变量。

    如果没有配置，就立即抛出容易理解的异常，
    避免程序运行到中途才报错。
    """
    value = os.getenv(name)

    if value is None or not value.strip():
        raise RuntimeError(
            f"缺少环境变量 {name}，请检查：{PROJECT_ROOT / '.env'}"
        )

    return value.strip()


def get_bool_env(name: str, default: bool = False) -> bool:
    """读取布尔类型环境变量，并拒绝拼写错误的配置值。"""
    value = os.getenv(name)

    if value is None:
        return default

    normalized = value.strip().lower()

    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"{name} 必须是 true/false、1/0、yes/no 或 on/off"
    )


def get_project_path_env(name: str, default: Path) -> Path:
    """读取路径配置；相对路径统一以项目根目录为基准。"""
    raw_value = os.getenv(name)

    if raw_value is None or not raw_value.strip():
        return default

    path = Path(raw_value.strip())

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    return path


def get_csv_env(name: str) -> tuple[str, ...]:
    """将英文逗号分隔的配置解析成元组。"""
    value = os.getenv(name, "")

    return tuple(
        item.strip().lower()
        for item in value.split(",")
        if item.strip()
    )


@dataclass(frozen=True)
class Settings:
    """
    客服 Agent 的统一配置。

    frozen=True 表示对象创建之后不能被修改，
    可以避免程序运行过程中意外修改配置。
    """

    # 大语言模型配置
    llm_api_key: str
    llm_base_url: str | None
    llm_model_name: str

    # 向量模型配置
    embedding_model_name: str

    # 项目路径
    knowledge_dir: Path
    database_path: Path
    database_url: str | None
    ingestion_report_path: Path

    # 生产级基础设施
    database_pool_size: int
    database_max_overflow: int
    vector_store_backend: str
    qdrant_url: str | None
    qdrant_api_key: str | None
    qdrant_collection: str
    qdrant_timeout_seconds: float
    enable_retrieval_cache: bool
    redis_url: str | None
    retrieval_cache_ttl_seconds: int

    # API服务配置
    app_env: str
    service_api_key: str | None
    api_max_concurrency: int
    api_request_timeout_seconds: float

    # 受控网页知识源配置
    enable_web_ingestion: bool
    web_sources_path: Path
    web_allowed_domains: tuple[str, ...]

    # 知识切片配置
    chunk_size: int
    chunk_overlap: int
    knowledge_min_content_chars: int

    # 知识库检索配置
    rag_top_k: int
    rag_min_score: float
    rag_keyword_weight: float
    rag_min_keyword_score: float
    rag_candidate_multiplier: int
    rag_enable_reranker: bool
    reranker_model_name: str
    rerank_candidate_k: int
    intent_min_confidence: float

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE必须大于0")
        if self.chunk_overlap < 0:
            raise ValueError("CHUNK_OVERLAP不能小于0")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP必须小于CHUNK_SIZE")
        if self.knowledge_min_content_chars <= 0:
            raise ValueError(
                "KNOWLEDGE_MIN_CONTENT_CHARS必须大于0"
            )
        if self.app_env not in {
            "development",
            "test",
            "production",
        }:
            raise ValueError(
                "APP_ENV必须是development、test或production"
            )
        if (
            self.app_env == "production"
            and not self.service_api_key
        ):
            raise ValueError(
                "生产环境必须配置SERVICE_API_KEY"
            )
        if self.api_max_concurrency <= 0:
            raise ValueError(
                "API_MAX_CONCURRENCY必须大于0"
            )
        if self.api_request_timeout_seconds <= 0:
            raise ValueError(
                "API_REQUEST_TIMEOUT_SECONDS必须大于0"
            )
        if self.database_pool_size <= 0:
            raise ValueError("DATABASE_POOL_SIZE必须大于0")
        if self.database_max_overflow < 0:
            raise ValueError(
                "DATABASE_MAX_OVERFLOW不能小于0"
            )
        if self.vector_store_backend not in {
            "memory",
            "qdrant",
        }:
            raise ValueError(
                "VECTOR_STORE_BACKEND必须是memory或qdrant"
            )
        if (
            self.vector_store_backend == "qdrant"
            and not self.qdrant_url
        ):
            raise ValueError(
                "使用Qdrant时必须配置QDRANT_URL"
            )
        if self.qdrant_timeout_seconds <= 0:
            raise ValueError(
                "QDRANT_TIMEOUT_SECONDS必须大于0"
            )
        if self.enable_retrieval_cache and not self.redis_url:
            raise ValueError(
                "启用检索缓存时必须配置REDIS_URL"
            )
        if self.retrieval_cache_ttl_seconds <= 0:
            raise ValueError(
                "RETRIEVAL_CACHE_TTL_SECONDS必须大于0"
            )
        if self.rag_top_k <= 0:
            raise ValueError("RAG_TOP_K必须大于0")
        if not -1.0 <= self.rag_min_score <= 1.0:
            raise ValueError("RAG_MIN_SCORE必须在-1到1之间")
        if not 0.0 <= self.rag_keyword_weight <= 1.0:
            raise ValueError(
                "RAG_KEYWORD_WEIGHT必须在0到1之间"
            )
        if not 0.0 <= self.rag_min_keyword_score <= 1.0:
            raise ValueError(
                "RAG_MIN_KEYWORD_SCORE必须在0到1之间"
            )
        if self.rag_candidate_multiplier <= 0:
            raise ValueError(
                "RAG_CANDIDATE_MULTIPLIER必须大于0"
            )
        if self.rerank_candidate_k <= 0:
            raise ValueError(
                "RERANK_CANDIDATE_K必须大于0"
            )
        if not 0.0 <= self.intent_min_confidence <= 1.0:
            raise ValueError(
                "INTENT_MIN_CONFIDENCE必须在0到1之间"
            )
        if self.enable_web_ingestion and not self.web_allowed_domains:
            raise ValueError(
                "ENABLE_WEB_INGESTION=true 时，"
                "WEB_ALLOWED_DOMAINS 不能为空"
            )

    def safe_dict(self) -> dict:
        """
        返回可以安全打印的配置。

        不打印API密钥，防止密钥泄露。
        """
        return {
            "llm_api_key_exists": bool(self.llm_api_key),
            "llm_base_url_configured": bool(self.llm_base_url),
            "llm_model_name": self.llm_model_name,
            "embedding_model_name": self.embedding_model_name,
            "knowledge_dir": str(self.knowledge_dir),
            "database_path": str(self.database_path),
            "database_url_configured": bool(
                self.database_url
            ),
            "database_pool_size": self.database_pool_size,
            "database_max_overflow": (
                self.database_max_overflow
            ),
            "vector_store_backend": (
                self.vector_store_backend
            ),
            "qdrant_url_configured": bool(self.qdrant_url),
            "qdrant_api_key_exists": bool(
                self.qdrant_api_key
            ),
            "qdrant_collection": self.qdrant_collection,
            "qdrant_timeout_seconds": (
                self.qdrant_timeout_seconds
            ),
            "enable_retrieval_cache": (
                self.enable_retrieval_cache
            ),
            "redis_url_configured": bool(self.redis_url),
            "retrieval_cache_ttl_seconds": (
                self.retrieval_cache_ttl_seconds
            ),
            "ingestion_report_path": str(
                self.ingestion_report_path
            ),
            "app_env": self.app_env,
            "service_api_key_exists": bool(
                self.service_api_key
            ),
            "api_max_concurrency": (
                self.api_max_concurrency
            ),
            "api_request_timeout_seconds": (
                self.api_request_timeout_seconds
            ),
            "enable_web_ingestion": self.enable_web_ingestion,
            "web_sources_path": str(self.web_sources_path),
            "web_allowed_domains": list(self.web_allowed_domains),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "knowledge_min_content_chars": (
                self.knowledge_min_content_chars
            ),
            "rag_top_k": self.rag_top_k,
            "rag_min_score": self.rag_min_score,
            "rag_keyword_weight": self.rag_keyword_weight,
            "rag_min_keyword_score": (
                self.rag_min_keyword_score
            ),
            "rag_candidate_multiplier": (
                self.rag_candidate_multiplier
            ),
            "rag_enable_reranker": (
                self.rag_enable_reranker
            ),
            "reranker_model_name": (
                self.reranker_model_name
            ),
            "rerank_candidate_k": (
                self.rerank_candidate_k
            ),
            "intent_min_confidence": self.intent_min_confidence,
        }


settings = Settings(
    llm_api_key=get_required_env("LLM_API_KEY"),

    # 如果使用第三方OpenAI兼容服务，需要配置base_url。
    # 如果直接使用官方接口，可以为空。
    llm_base_url=os.getenv("LLM_BASE_URL") or None,

    llm_model_name=get_required_env("LLM_MODEL_NAME"),

    embedding_model_name=os.getenv(
        "EMBEDDING_MODEL_NAME",
        "BAAI/bge-small-zh-v1.5",
    ),

    knowledge_dir=get_project_path_env(
        "KNOWLEDGE_DIR",
        PROJECT_ROOT / "knowledge",
    ),

    database_path=get_project_path_env(
        "DATABASE_PATH",
        PROJECT_ROOT / "data" / "customer_service.db",
    ),

    database_url=(
        os.getenv("DATABASE_URL") or None
    ),

    ingestion_report_path=get_project_path_env(
        "INGESTION_REPORT_PATH",
        PROJECT_ROOT / "data" / "ingestion_report.json",
    ),

    database_pool_size=int(
        os.getenv("DATABASE_POOL_SIZE", "5")
    ),

    database_max_overflow=int(
        os.getenv("DATABASE_MAX_OVERFLOW", "10")
    ),

    vector_store_backend=os.getenv(
        "VECTOR_STORE_BACKEND",
        "memory",
    ).strip().lower(),

    qdrant_url=os.getenv("QDRANT_URL") or None,

    qdrant_api_key=os.getenv("QDRANT_API_KEY") or None,

    qdrant_collection=os.getenv(
        "QDRANT_COLLECTION",
        "customer_service_knowledge",
    ).strip(),

    qdrant_timeout_seconds=float(
        os.getenv("QDRANT_TIMEOUT_SECONDS", "10")
    ),

    enable_retrieval_cache=get_bool_env(
        "ENABLE_RETRIEVAL_CACHE",
        default=False,
    ),

    redis_url=os.getenv("REDIS_URL") or None,

    retrieval_cache_ttl_seconds=int(
        os.getenv(
            "RETRIEVAL_CACHE_TTL_SECONDS",
            "300",
        )
    ),

    app_env=os.getenv(
        "APP_ENV",
        "development",
    ).strip().lower(),

    service_api_key=(
        os.getenv("SERVICE_API_KEY") or None
    ),

    api_max_concurrency=int(
        os.getenv("API_MAX_CONCURRENCY", "1")
    ),

    api_request_timeout_seconds=float(
        os.getenv(
            "API_REQUEST_TIMEOUT_SECONDS",
            "60",
        )
    ),

    enable_web_ingestion=get_bool_env(
        "ENABLE_WEB_INGESTION",
        default=False,
    ),

    web_sources_path=get_project_path_env(
        "WEB_SOURCES_PATH",
        PROJECT_ROOT / "knowledge" / "web_sources.json",
    ),

    web_allowed_domains=get_csv_env(
        "WEB_ALLOWED_DOMAINS"
    ),

    chunk_size=int(os.getenv("CHUNK_SIZE", "400")),

    chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "80")),

    knowledge_min_content_chars=int(
        os.getenv(
            "KNOWLEDGE_MIN_CONTENT_CHARS",
            "10",
        )
    ),

    rag_top_k=int(os.getenv("RAG_TOP_K", "3")),

    # 这是第一版的初始值，后面会使用评测集重新校准。
    rag_min_score=float(os.getenv("RAG_MIN_SCORE", "0.45")),

    rag_keyword_weight=float(
        os.getenv("RAG_KEYWORD_WEIGHT", "0.30")
    ),

    rag_min_keyword_score=float(
        os.getenv("RAG_MIN_KEYWORD_SCORE", "0.35")
    ),

    rag_candidate_multiplier=int(
        os.getenv("RAG_CANDIDATE_MULTIPLIER", "4")
    ),

    rag_enable_reranker=get_bool_env(
        "RAG_ENABLE_RERANKER",
        default=True,
    ),

    reranker_model_name=os.getenv(
        "RERANKER_MODEL_NAME",
        "BAAI/bge-reranker-base",
    ),

    rerank_candidate_k=int(
        os.getenv("RERANK_CANDIDATE_K", "10")
    ),

    # 低于该置信度的意图不会触发知识或业务工具。
    intent_min_confidence=float(
        os.getenv("INTENT_MIN_CONFIDENCE", "0.60")
    ),
)
