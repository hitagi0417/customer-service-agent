"""add token, tool and citation observability

Revision ID: 20260728_0002
Revises: 20260728_0001
Create Date: 2026-07-28
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "20260728_0002"
down_revision: str | None = "20260728_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "evaluation_records",
        sa.Column(
            "tool_calls",
            sa.JSON(),
            server_default="[]",
            nullable=False,
        ),
    )
    op.add_column(
        "evaluation_records",
        sa.Column(
            "cited_chunk_ids",
            sa.JSON(),
            server_default="[]",
            nullable=False,
        ),
    )
    op.add_column(
        "evaluation_records",
        sa.Column(
            "cited_sources",
            sa.JSON(),
            server_default="[]",
            nullable=False,
        ),
    )

    for column_name in (
        "model_calls",
        "usage_available_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_prompt_tokens",
    ):
        op.add_column(
            "evaluation_records",
            sa.Column(
                column_name,
                sa.Integer(),
                server_default="0",
                nullable=False,
            ),
        )

    op.add_column(
        "evaluation_records",
        sa.Column(
            "estimated_cost_usd",
            sa.Float(),
            server_default="0",
            nullable=False,
        ),
    )

    op.create_table(
        "tool_execution_records",
        sa.Column(
            "tool_call_id",
            sa.String(length=160),
            nullable=False,
        ),
        sa.Column(
            "request_id",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "call_index",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "tool_name",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "success",
            sa.Boolean(),
            nullable=False,
        ),
        sa.Column(
            "duration_ms",
            sa.Float(),
            nullable=False,
        ),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["evaluation_records.request_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("tool_call_id"),
    )
    op.create_index(
        "idx_tool_execution_request_id",
        "tool_execution_records",
        ["request_id"],
        unique=False,
    )
    op.create_index(
        "idx_tool_execution_name",
        "tool_execution_records",
        ["tool_name"],
        unique=False,
    )

    op.create_table(
        "citation_records",
        sa.Column(
            "citation_id",
            sa.String(length=180),
            nullable=False,
        ),
        sa.Column(
            "request_id",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "citation_index",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "chunk_id",
            sa.String(length=100),
            nullable=False,
        ),
        sa.Column(
            "source",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "is_retrieved",
            sa.Boolean(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["request_id"],
            ["evaluation_records.request_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("citation_id"),
    )
    op.create_index(
        "idx_citation_request_id",
        "citation_records",
        ["request_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "idx_citation_request_id",
        table_name="citation_records",
    )
    op.drop_table("citation_records")
    op.drop_index(
        "idx_tool_execution_name",
        table_name="tool_execution_records",
    )
    op.drop_index(
        "idx_tool_execution_request_id",
        table_name="tool_execution_records",
    )
    op.drop_table("tool_execution_records")

    for column_name in (
        "estimated_cost_usd",
        "cached_prompt_tokens",
        "total_tokens",
        "completion_tokens",
        "prompt_tokens",
        "usage_available_calls",
        "model_calls",
        "cited_sources",
        "cited_chunk_ids",
        "tool_calls",
    ):
        op.drop_column(
            "evaluation_records",
            column_name,
        )
