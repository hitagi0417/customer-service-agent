import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from app.config import settings
from app.ingestion import IngestionReport
from app.parsers import HtmlParser, ParsedSection, ParserRegistry, WebFetcher


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KnowledgeChunk:
    """
    一条已经切分完成的知识片段。

    这是知识库加载模块的输出，
    后面的向量检索模块会对content生成向量。
    """

    chunk_id: str
    source: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParsedDocument:
    """一个文件完成格式解析后的文档级结果。"""

    source: str
    doc_id: str
    version: str
    version_key: tuple[int, ...]
    effective_date: date | None
    sections: list[ParsedSection]


class KnowledgeLoader:
    """
    负责读取、清理和切分知识文件。
    """

    def __init__(
        self,
        knowledge_dir: Path,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        min_content_chars: int = 10,
        as_of_date: date | None = None,
        parser_registry: ParserRegistry | None = None,
    ) -> None:
        """
        初始化知识库加载器。

        knowledge_dir：
            知识文件所在目录。

        chunk_size：
            每个知识片段允许包含的最大字符数。

        chunk_overlap：
            相邻知识片段之间重复保留的字符数。
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size必须大于0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap不能小于0")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap必须小于chunk_size"
            )
        if min_content_chars <= 0:
            raise ValueError("min_content_chars必须大于0")

        self.knowledge_dir = knowledge_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_content_chars = min_content_chars
        self.as_of_date = as_of_date or date.today()
        self.parser_registry = parser_registry or ParserRegistry()
        self.report = IngestionReport()
        self._seen_file_hashes: dict[str, str] = {}
        self._seen_content_hashes: dict[str, str] = {}

    @property
    def supported_suffixes(self) -> set[str]:
        return self.parser_registry.supported_suffixes

    def load(
        self,
        allow_empty: bool = False,
    ) -> list[KnowledgeChunk]:
        """
        加载知识目录中的全部支持文件。

        返回值是一组KnowledgeChunk。
        """
        if not self.knowledge_dir.exists():
            raise FileNotFoundError(
                f"知识库目录不存在：{self.knowledge_dir}"
            )

        if not self.knowledge_dir.is_dir():
            raise NotADirectoryError(
                f"知识库路径不是文件夹：{self.knowledge_dir}"
            )

        file_paths = sorted(
            path
            for path in self.knowledge_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in self.supported_suffixes
        )

        self._reset_run_state()

        if not file_paths and not allow_empty:
            raise ValueError(
                f"知识库中没有找到支持的文档或代码文件："
                f"{self.knowledge_dir}"
            )

        chunks: list[KnowledgeChunk] = []
        parsed_documents: list[ParsedDocument] = []

        for file_path in file_paths:
            source = file_path.relative_to(
                self.knowledge_dir
            ).as_posix()
            self.report.total_files += 1

            try:
                file_hash = self._calculate_file_hash(
                    file_path
                )
                original_source = (
                    self._seen_file_hashes.get(file_hash)
                )

                if original_source is not None:
                    self.report.skipped_files += 1
                    self.report.duplicate_files += 1
                    self.report.add_issue(
                        source=source,
                        stage="deduplication",
                        reason="duplicate_file",
                        message=(
                            "文件内容与已有知识源完全相同"
                        ),
                        metadata={
                            "duplicate_of": original_source,
                            "file_hash": file_hash,
                        },
                    )
                    continue

                self._seen_file_hashes[file_hash] = source
                parsed_sections = self.parser_registry.parse(
                    path=file_path,
                    source=source,
                )
                sections = [
                    ParsedSection(
                        source=section.source,
                        content=section.content,
                        metadata={
                            **section.metadata,
                            "file_hash": file_hash,
                        },
                    )
                    for section in parsed_sections
                ]
                self.report.total_sections += len(sections)

                document = self._prepare_document(
                    source=source,
                    sections=sections,
                )

                if document is None:
                    self.report.skipped_files += 1
                    continue

                parsed_documents.append(document)

            except Exception as error:
                self.report.failed_files += 1
                self.report.add_issue(
                    source=source,
                    stage="parsing",
                    reason="parser_error",
                    message=(
                        f"{type(error).__name__}: {error}"
                    ),
                )
                logger.exception(
                    "知识文件解析失败，已隔离：%s",
                    file_path,
                )

        selected_documents = (
            self._select_latest_documents(
                parsed_documents
            )
        )

        for document in selected_documents:
            file_chunks = self._sections_to_chunks(
                document.sections
            )
            chunks.extend(file_chunks)

            if file_chunks:
                self.report.parsed_files += 1
            else:
                self.report.skipped_files += 1

        if not chunks and not allow_empty:
            raise ValueError("知识文件存在，但没有读取到有效内容")

        return chunks

    def _prepare_document(
        self,
        source: str,
        sections: list[ParsedSection],
    ) -> ParsedDocument | None:
        """校验文档级状态、版本和生效日期。"""
        if not sections:
            self.report.filtered_sections += 1
            self.report.add_issue(
                source=source,
                stage="document_filter",
                reason="empty_document",
                message="解析器没有返回任何内容",
            )
            return None

        metadata = sections[0].metadata
        doc_id = str(
            metadata.get("doc_id") or source
        ).strip()
        status = str(
            metadata.get("status", "active")
        ).strip().lower()

        if status in {
            "inactive",
            "draft",
            "archived",
        }:
            self.report.inactive_documents += 1
            self.report.add_issue(
                source=source,
                stage="document_filter",
                reason=f"status_{status}",
                message=(
                    f"文档状态为{status}，不进入当前索引"
                ),
                metadata={
                    "doc_id": doc_id,
                    "status": status,
                },
            )
            return None

        if status != "active":
            self.report.invalid_metadata_documents += 1
            self.report.add_issue(
                source=source,
                stage="metadata_validation",
                reason="invalid_status",
                message=f"不支持的文档状态：{status}",
                metadata={
                    "doc_id": doc_id,
                    "status": status,
                },
            )
            return None

        effective_date = self._parse_effective_date(
            source=source,
            doc_id=doc_id,
            raw_value=metadata.get(
                "effective_date"
            ),
        )

        if (
            metadata.get("effective_date")
            and effective_date is None
        ):
            return None

        if (
            effective_date is not None
            and effective_date > self.as_of_date
        ):
            self.report.future_documents += 1
            self.report.add_issue(
                source=source,
                stage="document_filter",
                reason="not_yet_effective",
                message=(
                    f"文档将在{effective_date.isoformat()}生效"
                ),
                metadata={
                    "doc_id": doc_id,
                    "effective_date": (
                        effective_date.isoformat()
                    ),
                    "as_of_date": (
                        self.as_of_date.isoformat()
                    ),
                },
            )
            return None

        version = str(
            metadata.get("version", "0")
        ).strip()
        version_key = self._parse_version(
            source=source,
            doc_id=doc_id,
            version=version,
        )

        if version_key is None:
            return None

        return ParsedDocument(
            source=source,
            doc_id=doc_id,
            version=version,
            version_key=version_key,
            effective_date=effective_date,
            sections=sections,
        )

    def _parse_effective_date(
        self,
        source: str,
        doc_id: str,
        raw_value: Any,
    ) -> date | None:
        if raw_value is None or not str(raw_value).strip():
            return None

        try:
            return date.fromisoformat(
                str(raw_value).strip()
            )
        except ValueError:
            self.report.invalid_metadata_documents += 1
            self.report.add_issue(
                source=source,
                stage="metadata_validation",
                reason="invalid_effective_date",
                message=(
                    "effective_date必须使用YYYY-MM-DD格式"
                ),
                metadata={
                    "doc_id": doc_id,
                    "effective_date": str(raw_value),
                },
            )
            return None

    def _parse_version(
        self,
        source: str,
        doc_id: str,
        version: str,
    ) -> tuple[int, ...] | None:
        try:
            parts = tuple(
                int(part)
                for part in version.split(".")
            )

            if not parts or any(
                part < 0
                for part in parts
            ):
                raise ValueError

        except ValueError:
            self.report.invalid_metadata_documents += 1
            self.report.add_issue(
                source=source,
                stage="metadata_validation",
                reason="invalid_version",
                message=(
                    "version必须是由点分隔的非负整数"
                ),
                metadata={
                    "doc_id": doc_id,
                    "version": version,
                },
            )
            return None

        normalized = list(parts)

        while (
            len(normalized) > 1
            and normalized[-1] == 0
        ):
            normalized.pop()

        return tuple(normalized)

    def _select_latest_documents(
        self,
        documents: list[ParsedDocument],
    ) -> list[ParsedDocument]:
        """同一doc_id只选择当前日期下版本最高的有效文档。"""
        grouped: dict[
            str,
            list[ParsedDocument],
        ] = {}

        for document in documents:
            grouped.setdefault(
                document.doc_id,
                [],
            ).append(document)

        selected: list[ParsedDocument] = []

        for doc_id, candidates in grouped.items():
            latest = max(
                candidates,
                key=lambda document: (
                    document.version_key,
                    document.effective_date
                    or date.min,
                    document.source,
                ),
            )
            selected.append(latest)

            for candidate in candidates:
                if candidate is latest:
                    continue

                self.report.skipped_files += 1
                self.report.superseded_documents += 1
                self.report.add_issue(
                    source=candidate.source,
                    stage="version_selection",
                    reason="superseded_version",
                    message=(
                        f"同一doc_id已选择更高版本"
                        f"{latest.version}"
                    ),
                    metadata={
                        "doc_id": doc_id,
                        "version": candidate.version,
                        "selected_source": latest.source,
                        "selected_version": latest.version,
                    },
                )

        return sorted(
            selected,
            key=lambda document: document.source,
        )

    def _reset_run_state(self) -> None:
        """开始一次新的知识库构建，清空上一次的统计和去重状态。"""
        self.report = IngestionReport()
        self._seen_file_hashes.clear()
        self._seen_content_hashes.clear()

    def _calculate_file_hash(
        self,
        path: Path,
    ) -> str:
        """流式计算文件哈希，避免大文件一次性读入内存。"""
        digest = hashlib.sha256()

        with path.open("rb") as file:
            while block := file.read(1024 * 1024):
                digest.update(block)

        return digest.hexdigest()

    def save_report(self, path: Path) -> None:
        self.report.total_chunks = len(
            self._seen_content_hashes
        )
        self.report.save(path)

    def load_web_urls(
        self,
        urls: list[str],
        allowed_domains: set[str],
    ) -> list[KnowledgeChunk]:
        """抓取管理员配置的网页并转换为知识片段。"""
        if not urls:
            return []

        fetcher = WebFetcher(allowed_domains=allowed_domains)
        parser = HtmlParser()
        chunks: list[KnowledgeChunk] = []

        for url in urls:
            self.report.total_web_pages += 1

            try:
                html = fetcher.fetch(url)
                sections = parser.parse_html(
                    html=html,
                    source=url,
                    source_url=url,
                )
                self.report.total_sections += len(
                    sections
                )
                page_chunks = self._sections_to_chunks(
                    sections
                )
                chunks.extend(page_chunks)

                if page_chunks:
                    self.report.parsed_web_pages += 1
                else:
                    self.report.skipped_web_pages += 1

            except Exception as error:
                self.report.failed_web_pages += 1
                self.report.add_issue(
                    source=url,
                    stage="web_fetch_or_parse",
                    reason="web_source_error",
                    message=(
                        f"{type(error).__name__}: {error}"
                    ),
                )
                logger.exception(
                    "网页知识源处理失败，已隔离：%s",
                    url,
                )

        return chunks

    def _sections_to_chunks(
        self,
        sections: list[ParsedSection],
    ) -> list[KnowledgeChunk]:
        """将带语义边界的解析结果切成最终知识片段。"""
        chunks: list[KnowledgeChunk] = []

        for section_index, section in enumerate(
            sections,
            start=1,
        ):
            parse_status = section.metadata.get(
                "parse_status",
                "parsed",
            )

            if parse_status != "parsed":
                self.report.filtered_sections += 1

                if parse_status == "ocr_required":
                    self.report.ocr_required_pages += 1

                self.report.add_issue(
                    source=section.source,
                    stage="parsing",
                    reason=str(parse_status),
                    message=(
                        "页面包含图片但没有可提取文本，需要OCR"
                        if parse_status == "ocr_required"
                        else "页面没有可提取文本"
                    ),
                    metadata=dict(section.metadata),
                )
                continue

            cleaned_text = self._clean_text(section.content)
            invalid_reason = self._get_invalid_content_reason(
                cleaned_text
            )

            if invalid_reason is not None:
                self.report.filtered_sections += 1
                self.report.add_issue(
                    source=section.source,
                    stage="quality_filter",
                    reason=invalid_reason,
                    metadata={
                        **section.metadata,
                        "content_length": len(cleaned_text),
                    },
                )
                continue

            section_chunks = self._split_text(cleaned_text)

            for chunk_index, content in enumerate(
                section_chunks,
                start=1,
            ):
                metadata = dict(section.metadata)
                metadata.setdefault("doc_id", section.source)
                metadata["section_index"] = section_index
                metadata["chunk_index"] = chunk_index
                full_content_hash = hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest()
                metadata["content_hash"] = (
                    full_content_hash[:16]
                )

                duplicate_of = (
                    self._seen_content_hashes.get(
                        full_content_hash
                    )
                )

                if duplicate_of is not None:
                    self.report.duplicate_chunks += 1
                    self.report.add_issue(
                        source=section.source,
                        stage="deduplication",
                        reason="duplicate_chunk",
                        message=(
                            "清洗后的片段内容与已有片段相同"
                        ),
                        metadata={
                            "duplicate_of": duplicate_of,
                            "content_hash": (
                                full_content_hash[:16]
                            ),
                            "section_index": section_index,
                            "chunk_index": chunk_index,
                        },
                    )
                    continue

                self._seen_content_hashes[
                    full_content_hash
                ] = section.source

                chunk_id = self._create_chunk_id(
                    source=section.source,
                    content=content,
                    metadata=metadata,
                )

                chunks.append(
                    KnowledgeChunk(
                        chunk_id=chunk_id,
                        source=section.source,
                        content=content,
                        metadata=metadata,
                    )
                )
                self.report.total_chunks += 1

        return chunks

    def _get_invalid_content_reason(
        self,
        content: str,
    ) -> str | None:
        """返回内容被过滤的原因；返回None表示内容合格。"""
        if not content:
            return "empty_content"

        if len(content) < self.min_content_chars:
            return "content_too_short"

        if not any(
            character.isalnum()
            for character in content
        ):
            return "no_alphanumeric_content"

        return None

    def _clean_text(self, text: str) -> str:
        """
        清理文本格式。

        主要处理：
        1. 统一换行符；
        2. 删除多余空格；
        3. 删除过多空行；
        4. 删除开头和结尾的空白。
        """
        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # 连续的空格或Tab压缩成一个空格
        text = re.sub(r"[ \t]+", " ", text)

        # 三个及以上换行压缩成两个换行
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _split_text(self, text: str) -> list[str]:
        """
        按字符长度切分文本。

        切片时优先在换行、句号、问号等位置结束，
        尽量避免把一句话从中间截断。
        """
        chunks: list[str] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(
                start + self.chunk_size,
                text_length,
            )

            # 如果还没有到文本末尾，尝试寻找更自然的边界
            if end < text_length:
                search_start = start + self.chunk_size // 2

                boundary_positions = [
                    text.rfind(separator, search_start, end)
                    for separator in (
                        "\n",
                        "。",
                        "！",
                        "？",
                        "；",
                    )
                ]

                best_boundary = max(boundary_positions)

                if best_boundary > start:
                    # 加1是为了保留句号或换行符
                    end = best_boundary + 1

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end >= text_length:
                break

            # 下一个片段向前重叠一部分内容
            next_start = end - self.chunk_overlap

            # 防止特殊情况下无法向后移动，造成死循环
            if next_start <= start:
                next_start = end

            start = next_start

        return chunks

    def _create_chunk_id(
        self,
        source: str,
        content: str,
        metadata: dict[str, Any],
    ) -> str:
        """
        根据来源和内容生成稳定的知识片段ID。

        相同文件中的相同内容会生成相同ID。
        内容变化后，ID也会变化。
        """
        stable_metadata = {
            key: value
            for key, value in metadata.items()
            if key not in {"fetched_at"}
        }

        raw_value = "\n".join(
            (
                source,
                json.dumps(
                    stable_metadata,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
                content,
            )
        )

        digest = hashlib.sha1(
            raw_value.encode("utf-8")
        ).hexdigest()

        return digest[:16]


def load_knowledge_chunks() -> list[KnowledgeChunk]:
    """
    使用项目统一配置加载知识库。

    其他模块不需要重复创建KnowledgeLoader，
    直接调用这个函数即可。
    """
    loader = KnowledgeLoader(
        knowledge_dir=settings.knowledge_dir,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        min_content_chars=(
            settings.knowledge_min_content_chars
        ),
    )

    chunks: list[KnowledgeChunk] = []

    try:
        chunks.extend(
            loader.load(
                allow_empty=(
                    settings.enable_web_ingestion
                )
            )
        )

        if settings.enable_web_ingestion:
            web_urls = _load_web_source_urls(
                settings.web_sources_path
            )
            chunks.extend(
                loader.load_web_urls(
                    urls=web_urls,
                    allowed_domains=set(
                        settings.web_allowed_domains
                    ),
                )
            )

        if not chunks:
            raise ValueError(
                "所有知识源均未产生有效片段，"
                "请查看知识入库质量报告"
            )

        return chunks

    finally:
        loader.save_report(
            settings.ingestion_report_path
        )
        logger.info(
            (
                "知识入库统计：文件=%d，成功=%d，失败=%d，"
                "跳过=%d，片段=%d，重复文件=%d，"
                "重复片段=%d，待OCR页=%d，"
                "非活动文档=%d，未生效文档=%d，"
                "旧版本=%d"
            ),
            loader.report.total_files,
            loader.report.parsed_files,
            loader.report.failed_files,
            loader.report.skipped_files,
            loader.report.total_chunks,
            loader.report.duplicate_files,
            loader.report.duplicate_chunks,
            loader.report.ocr_required_pages,
            loader.report.inactive_documents,
            loader.report.future_documents,
            loader.report.superseded_documents,
        )
        logger.info(
            "知识入库质量报告已保存：%s",
            settings.ingestion_report_path,
        )


def _load_web_source_urls(path: Path) -> list[str]:
    """读取管理员维护的网页知识源清单。"""
    if not path.exists():
        raise FileNotFoundError(
            f"网页知识源清单不存在：{path}"
        )

    data = json.loads(path.read_text(encoding="utf-8-sig"))

    if not isinstance(data, list):
        raise ValueError("网页知识源清单必须是JSON数组")

    urls: list[str] = []

    for item in data:
        if isinstance(item, str):
            urls.append(item)
            continue

        if isinstance(item, dict) and isinstance(
            item.get("url"),
            str,
        ):
            urls.append(item["url"])
            continue

        raise ValueError(
            "网页知识源必须是URL字符串或包含url字段的对象"
        )

    return urls
