from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import (
    insert as postgresql_insert,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.database import (
    database_transaction,
    get_engine,
    initialize_database,
    service_tickets,
)


@dataclass(frozen=True)
class Ticket:
    ticket_id: str
    request_id: str
    question: str
    reason: str
    status: str
    created_at: datetime


class TicketRepository:
    """跨SQLite和PostgreSQL的工单仓储。"""

    def __init__(self) -> None:
        initialize_database()

    def create_or_get(
        self,
        ticket: Ticket,
    ) -> tuple[Ticket, bool]:
        """
        按request_id幂等创建工单。

        数据库唯一约束是最后一道防线，即使请求重试或并发到达，
        也只会保存一张工单。
        """
        dialect = get_engine().dialect.name
        values = {
            "ticket_id": ticket.ticket_id,
            "request_id": ticket.request_id,
            "question": ticket.question,
            "reason": ticket.reason,
            "status": ticket.status,
            "created_at": ticket.created_at,
        }

        if dialect == "postgresql":
            statement = (
                postgresql_insert(service_tickets)
                .values(**values)
                .on_conflict_do_nothing(
                    index_elements=["request_id"]
                )
                .returning(
                    service_tickets.c.ticket_id
                )
            )
        else:
            statement = (
                sqlite_insert(service_tickets)
                .values(**values)
                .on_conflict_do_nothing(
                    index_elements=["request_id"]
                )
                .returning(
                    service_tickets.c.ticket_id
                )
            )

        with database_transaction() as connection:
            result = connection.execute(statement)
            created = result.scalar_one_or_none() is not None
            row = (
                connection.execute(
                    select(service_tickets).where(
                        service_tickets.c.request_id
                        == ticket.request_id
                    )
                )
                .mappings()
                .first()
            )

        if row is None:
            raise RuntimeError("创建工单后无法查询到工单")

        created_at = row["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        return (
            Ticket(
                ticket_id=row["ticket_id"],
                request_id=row["request_id"],
                question=row["question"],
                reason=row["reason"],
                status=row["status"],
                created_at=created_at,
            ),
            created,
        )
