import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class IngestionIssue:
    """知识入库过程中被隔离、过滤或去重的一条问题记录。"""

    source: str
    stage: str
    reason: str
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestionReport:
    """一次知识库构建的数据质量报告。"""

    started_at: datetime = field(default_factory=utc_now)
    completed_at: datetime | None = None
    duration_ms: float = 0.0

    total_files: int = 0
    parsed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0

    total_web_pages: int = 0
    parsed_web_pages: int = 0
    failed_web_pages: int = 0
    skipped_web_pages: int = 0

    total_sections: int = 0
    filtered_sections: int = 0
    total_chunks: int = 0
    duplicate_files: int = 0
    duplicate_chunks: int = 0
    ocr_required_pages: int = 0
    inactive_documents: int = 0
    future_documents: int = 0
    superseded_documents: int = 0
    invalid_metadata_documents: int = 0

    issues: list[IngestionIssue] = field(default_factory=list)

    def add_issue(
        self,
        source: str,
        stage: str,
        reason: str,
        message: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.issues.append(
            IngestionIssue(
                source=source,
                stage=stage,
                reason=reason,
                message=message,
                metadata=metadata or {},
            )
        )

    def finish(self) -> None:
        self.completed_at = utc_now()
        self.duration_ms = round(
            (
                self.completed_at
                - self.started_at
            ).total_seconds()
            * 1000,
            2,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["started_at"] = self.started_at.isoformat()
        data["completed_at"] = (
            self.completed_at.isoformat()
            if self.completed_at
            else None
        )
        return data

    def save(self, path: Path) -> None:
        """将报告保存成便于人工排查和后续监控采集的JSON。"""
        if self.completed_at is None:
            self.finish()

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        path.write_text(
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
