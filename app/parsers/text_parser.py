from pathlib import Path

from app.parsers.base import ParsedSection, read_text_file


class TextParser:
    """普通文本解析器。"""

    def parse(
        self,
        path: Path,
        source: str,
    ) -> list[ParsedSection]:
        content = read_text_file(path).strip()

        if not content:
            return []

        return [
            ParsedSection(
                source=source,
                content=content,
                metadata={
                    "document_type": "text",
                },
            )
        ]
