from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class ParsedSection:
    """解析器输出的一段结构化内容。"""

    source: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class DocumentParser(Protocol):
    """所有本地文档解析器都遵循的接口。"""

    def parse(
        self,
        path: Path,
        source: str,
    ) -> list[ParsedSection]:
        ...


def read_text_file(path: Path) -> str:
    """兼容常见中文编码读取文本文件。"""
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise UnicodeError(f"无法识别文件编码：{path}")
