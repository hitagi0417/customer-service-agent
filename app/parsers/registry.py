from pathlib import Path

from app.parsers.base import DocumentParser, ParsedSection
from app.parsers.code_parser import GenericCodeParser, PythonCodeParser
from app.parsers.html_parser import HtmlParser
from app.parsers.markdown_parser import MarkdownParser
from app.parsers.pdf_parser import PdfParser
from app.parsers.text_parser import TextParser


class ParserRegistry:
    """根据文件扩展名选择解析器。"""

    def __init__(self) -> None:
        self._parsers: dict[str, DocumentParser] = {
            ".txt": TextParser(),
            ".md": MarkdownParser(),
            ".markdown": MarkdownParser(),
            ".pdf": PdfParser(),
            ".html": HtmlParser(),
            ".htm": HtmlParser(),
            ".py": PythonCodeParser(),
            ".js": GenericCodeParser("javascript"),
            ".jsx": GenericCodeParser("javascript"),
            ".ts": GenericCodeParser("typescript"),
            ".tsx": GenericCodeParser("typescript"),
            ".java": GenericCodeParser("java"),
            ".go": GenericCodeParser("go"),
            ".rs": GenericCodeParser("rust"),
            ".c": GenericCodeParser("c"),
            ".h": GenericCodeParser("c"),
            ".cpp": GenericCodeParser("cpp"),
            ".hpp": GenericCodeParser("cpp"),
            ".cs": GenericCodeParser("csharp"),
            ".php": GenericCodeParser("php"),
            ".rb": GenericCodeParser("ruby"),
            ".sh": GenericCodeParser("shell"),
            ".sql": GenericCodeParser("sql"),
        }

    @property
    def supported_suffixes(self) -> set[str]:
        return set(self._parsers)

    def parse(
        self,
        path: Path,
        source: str | None = None,
    ) -> list[ParsedSection]:
        parser = self._parsers.get(path.suffix.lower())

        if parser is None:
            raise ValueError(f"不支持的文件类型：{path.suffix}")

        return parser.parse(
            path=path,
            source=source or path.name,
        )
