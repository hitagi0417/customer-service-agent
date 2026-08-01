import sqlite3
from sqlite3 import Connection

from app.config import settings


def get_connection() -> Connection:
    """
    创建一个新的SQLite数据库连接。

    每次数据库操作都创建独立连接，
    不在整个程序中共享同一个连接。
    """
    # 确保data目录存在
    settings.database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        settings.database_path,
        timeout=10.0,
    )

    # 查询结果可以通过字段名访问
    connection.row_factory = sqlite3.Row

    # 启用外键约束
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def initialize_database() -> None:
    """
    初始化客服系统数据库。

    如果数据表不存在就创建；
    如果已经存在则不会重复创建或删除原数据。
    """
    connection = get_connection()

    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS service_tickets (
                ticket_id TEXT PRIMARY KEY,
                request_id TEXT NOT NULL,
                question TEXT NOT NULL,
                reason TEXT NOT NULL,

                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK (
                        status IN (
                            'pending',
                            'processing',
                            'resolved',
                            'closed'
                        )
                    ),

                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS
                idx_service_tickets_request_id
            ON service_tickets(request_id);

            CREATE UNIQUE INDEX IF NOT EXISTS
                uq_service_tickets_request_id
            ON service_tickets(request_id);

            CREATE INDEX IF NOT EXISTS
                idx_service_tickets_status
            ON service_tickets(status);

            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversation_messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL
                    CHECK (role IN ('user', 'assistant', 'tool')),
                content TEXT NOT NULL,
                token_estimate INTEGER NOT NULL
                    CHECK (token_estimate >= 0),
                summarized INTEGER NOT NULL DEFAULT 0
                    CHECK (summarized IN (0, 1)),
                created_at TEXT NOT NULL,
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(conversation_id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS
                idx_messages_conversation
            ON conversation_messages(
                conversation_id,
                message_id
            );

            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                customer_id TEXT NOT NULL,
                item_name TEXT NOT NULL,
                amount REAL NOT NULL CHECK (amount >= 0),
                status TEXT NOT NULL
                    CHECK (
                        status IN (
                            'paid',
                            'shipped',
                            'delivered',
                            'cancelled',
                            'refunded'
                        )
                    ),
                paid_at TEXT,
                shipped_at TEXT,
                delivered_at TEXT,
                refundable_days INTEGER NOT NULL DEFAULT 7
                    CHECK (refundable_days > 0)
            );

            CREATE INDEX IF NOT EXISTS idx_orders_customer
            ON orders(customer_id, order_id);

            INSERT OR IGNORE INTO orders (
                order_id,
                customer_id,
                item_name,
                amount,
                status,
                paid_at,
                shipped_at,
                delivered_at,
                refundable_days
            ) VALUES (
                'DEMO-1001',
                'demo_customer',
                'XJ-900 标准版',
                899.00,
                'delivered',
                datetime('now', '-6 days'),
                datetime('now', '-5 days'),
                datetime('now', '-3 days'),
                7
            );

            INSERT OR IGNORE INTO orders (
                order_id,
                customer_id,
                item_name,
                amount,
                status,
                paid_at,
                shipped_at,
                delivered_at,
                refundable_days
            ) VALUES (
                'DEMO-1002',
                'demo_customer',
                'A100',
                399.00,
                'shipped',
                datetime('now', '-2 days'),
                datetime('now', '-1 day'),
                NULL,
                7
            );


            CREATE TABLE IF NOT EXISTS evaluation_records (
                request_id TEXT PRIMARY KEY,

                question TEXT NOT NULL,

                predicted_intent TEXT NOT NULL
                    CHECK (
                        predicted_intent IN (
                            'knowledge_query',
                            'service_request',
                            'chitchat',
                            'unknown'
                        )
                    ),

                intent_confidence REAL NOT NULL
                    CHECK (
                        intent_confidence >= 0.0
                        AND intent_confidence <= 1.0
                    ),

                intent_reason TEXT NOT NULL,

                retrieved_chunk_ids TEXT NOT NULL
                    DEFAULT '[]',

                retrieval_scores TEXT NOT NULL
                    DEFAULT '[]',

                tool_names TEXT NOT NULL
                    DEFAULT '[]',

                answer TEXT NOT NULL,

                need_human INTEGER NOT NULL
                    CHECK (need_human IN (0, 1)),

                success INTEGER NOT NULL
                    CHECK (success IN (0, 1)),

                auto_resolved INTEGER NOT NULL DEFAULT 0
                    CHECK (auto_resolved IN (0, 1)),

                error TEXT,

                total_duration_ms REAL NOT NULL
                    CHECK (total_duration_ms >= 0),

                trace_metadata TEXT NOT NULL DEFAULT '{}',

                user_feedback INTEGER
                    CHECK (
                        user_feedback IS NULL
                        OR user_feedback IN (-1, 1)
                    ),

                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS
                idx_evaluation_records_intent
            ON evaluation_records(predicted_intent);

            CREATE INDEX IF NOT EXISTS
                idx_evaluation_records_created_at
            ON evaluation_records(created_at);
            """
        )

        evaluation_columns = {
            row["name"]
            for row in connection.execute(
                "PRAGMA table_info(evaluation_records)"
            ).fetchall()
        }

        if "auto_resolved" not in evaluation_columns:
            connection.execute(
                """
                ALTER TABLE evaluation_records
                ADD COLUMN auto_resolved INTEGER NOT NULL DEFAULT 0
                    CHECK (auto_resolved IN (0, 1))
                """
            )

        if "trace_metadata" not in evaluation_columns:
            connection.execute(
                """
                ALTER TABLE evaluation_records
                ADD COLUMN trace_metadata TEXT NOT NULL DEFAULT '{}'
                """
            )

        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()
