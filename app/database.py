import threading
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.engine import Connection, Engine

from app.config import settings


metadata = MetaData()


service_tickets = Table(
    "service_tickets",
    metadata,
    Column("ticket_id", String(64), primary_key=True),
    Column(
        "request_id",
        String(100),
        nullable=False,
        unique=True,
    ),
    Column("question", Text, nullable=False),
    Column("reason", Text, nullable=False),
    Column(
        "status",
        String(32),
        nullable=False,
        default="pending",
    ),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
    ),
    CheckConstraint(
        "status IN ('pending', 'processing', 'resolved', 'closed')",
        name="ck_service_tickets_status",
    ),
)
Index(
    "idx_service_tickets_request_id",
    service_tickets.c.request_id,
)
Index(
    "idx_service_tickets_status",
    service_tickets.c.status,
)


evaluation_records = Table(
    "evaluation_records",
    metadata,
    Column(
        "request_id",
        String(100),
        primary_key=True,
    ),
    Column("question", Text, nullable=False),
    Column(
        "predicted_intent",
        String(32),
        nullable=False,
    ),
    Column(
        "intent_confidence",
        Float,
        nullable=False,
    ),
    Column("intent_reason", Text, nullable=False),
    Column(
        "retrieved_chunk_ids",
        JSON,
        nullable=False,
        default=list,
    ),
    Column(
        "retrieval_scores",
        JSON,
        nullable=False,
        default=list,
    ),
    Column(
        "tool_names",
        JSON,
        nullable=False,
        default=list,
    ),
    Column("answer", Text, nullable=False),
    Column("need_human", Boolean, nullable=False),
    Column("success", Boolean, nullable=False),
    Column(
        "auto_resolved",
        Boolean,
        nullable=False,
        default=False,
    ),
    Column("error", Text),
    Column(
        "total_duration_ms",
        Float,
        nullable=False,
    ),
    Column("user_feedback", Integer),
    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
    ),
    CheckConstraint(
        (
            "predicted_intent IN "
            "('knowledge_query', 'service_request', 'chitchat', 'unknown')"
        ),
        name="ck_evaluation_records_intent",
    ),
    CheckConstraint(
        "intent_confidence >= 0.0 AND intent_confidence <= 1.0",
        name="ck_evaluation_records_confidence",
    ),
    CheckConstraint(
        "total_duration_ms >= 0",
        name="ck_evaluation_records_duration",
    ),
    CheckConstraint(
        (
            "user_feedback IS NULL "
            "OR user_feedback IN (-1, 1)"
        ),
        name="ck_evaluation_records_feedback",
    ),
)
Index(
    "idx_evaluation_records_intent",
    evaluation_records.c.predicted_intent,
)
Index(
    "idx_evaluation_records_created_at",
    evaluation_records.c.created_at,
)


_engines: dict[str, Engine] = {}
_engines_lock = threading.Lock()


def get_database_url() -> str:
    """
    返回当前数据库连接串。

    DATABASE_URL存在时使用PostgreSQL等外部数据库；
    否则根据DATABASE_PATH生成SQLite连接串，供本地开发和测试使用。
    """
    if settings.database_url:
        return settings.database_url

    settings.database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path = settings.database_path.resolve().as_posix()
    return f"sqlite+pysqlite:///{path}"


def get_engine() -> Engine:
    """
    按连接串复用SQLAlchemy连接池。

    测试会动态替换SQLite文件路径，因此这里按URL缓存，
    而不是在模块导入时固定创建一个全局连接。
    """
    database_url = get_database_url()

    with _engines_lock:
        cached = _engines.get(database_url)

        if cached is not None:
            return cached

        if database_url.startswith("sqlite"):
            engine = create_engine(
                database_url,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 10,
                },
                pool_pre_ping=True,
            )
        else:
            engine = create_engine(
                database_url,
                pool_size=settings.database_pool_size,
                max_overflow=(
                    settings.database_max_overflow
                ),
                pool_pre_ping=True,
                pool_recycle=1800,
            )

        _engines[database_url] = engine
        return engine


@contextmanager
def database_transaction() -> Iterator[Connection]:
    """创建自动提交或回滚的数据库事务。"""
    with get_engine().begin() as connection:
        yield connection


def initialize_database() -> None:
    """创建业务表，并兼容项目早期版本的SQLite文件。"""
    engine = get_engine()
    metadata.create_all(engine)

    if engine.dialect.name != "sqlite":
        return

    columns = {
        column["name"]
        for column in inspect(engine).get_columns(
            "evaluation_records"
        )
    }

    if "auto_resolved" not in columns:
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    ALTER TABLE evaluation_records
                    ADD COLUMN auto_resolved
                    INTEGER NOT NULL DEFAULT 0
                    CHECK (auto_resolved IN (0, 1))
                    """
                )
            )


def check_database_health() -> bool:
    """执行最小查询，供健康检查确认数据库真实可用。"""
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def dispose_database_engines() -> None:
    """释放连接池，主要供测试和进程优雅退出使用。"""
    with _engines_lock:
        engines = list(_engines.values())
        _engines.clear()

    for engine in engines:
        engine.dispose()
