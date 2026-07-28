"""create customer service business tables

Revision ID: 20260728_0001
Revises:
Create Date: 2026-07-28
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260728_0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "service_tickets",
        sa.Column(
            "ticket_id",
            sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            "request_id",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "question",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "reason",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=32),
            server_default="pending",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            (
                "status IN "
                "('pending', 'processing', 'resolved', 'closed')"
            ),
            name="ck_service_tickets_status",
        ),
        sa.PrimaryKeyConstraint("ticket_id"),
        sa.UniqueConstraint(
            "request_id",
            name="uq_service_tickets_request_id",
        ),
    )
    op.create_index(
        "idx_service_tickets_request_id",
        "service_tickets",
        ["request_id"],
        unique=False,
    )
    op.create_index(
        "idx_service_tickets_status",
        "service_tickets",
        ["status"],
        unique=False,
    )

    op.create_table(
        "evaluation_records",
        sa.Column(
            "request_id",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "question",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "predicted_intent",
            sa.String(length=32),
            nullable=False,
        ),
        sa.Column(
            "intent_confidence",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "intent_reason",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "retrieved_chunk_ids",
            sa.JSON(),
            nullable=False,
        ),
        sa.Column(
            "retrieval_scores",
            sa.JSON(),
            nullable=False,
        ),
        sa.Column(
            "tool_names",
            sa.JSON(),
            nullable=False,
        ),
        sa.Column(
            "answer",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "need_human",
            sa.Boolean(),
            nullable=False,
        ),
        sa.Column(
            "success",
            sa.Boolean(),
            nullable=False,
        ),
        sa.Column(
            "auto_resolved",
            sa.Boolean(),
            nullable=False,
        ),
        sa.Column(
            "error",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "total_duration_ms",
            sa.Float(),
            nullable=False,
        ),
        sa.Column(
            "user_feedback",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.CheckConstraint(
            (
                "predicted_intent IN "
                "('knowledge_query', 'service_request', "
                "'chitchat', 'unknown')"
            ),
            name="ck_evaluation_records_intent",
        ),
        sa.CheckConstraint(
            (
                "intent_confidence >= 0.0 "
                "AND intent_confidence <= 1.0"
            ),
            name="ck_evaluation_records_confidence",
        ),
        sa.CheckConstraint(
            "total_duration_ms >= 0",
            name="ck_evaluation_records_duration",
        ),
        sa.CheckConstraint(
            (
                "user_feedback IS NULL "
                "OR user_feedback IN (-1, 1)"
            ),
            name="ck_evaluation_records_feedback",
        ),
        sa.PrimaryKeyConstraint("request_id"),
    )
    op.create_index(
        "idx_evaluation_records_created_at",
        "evaluation_records",
        ["created_at"],
        unique=False,
    )
    op.create_index(
        "idx_evaluation_records_intent",
        "evaluation_records",
        ["predicted_intent"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "idx_evaluation_records_intent",
        table_name="evaluation_records",
    )
    op.drop_index(
        "idx_evaluation_records_created_at",
        table_name="evaluation_records",
    )
    op.drop_table("evaluation_records")
    op.drop_index(
        "idx_service_tickets_status",
        table_name="service_tickets",
    )
    op.drop_index(
        "idx_service_tickets_request_id",
        table_name="service_tickets",
    )
    op.drop_table("service_tickets")
