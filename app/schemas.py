from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """
    返回带时区的UTC时间。

    数据库存储UTC时间，可以避免服务器部署到不同时区后，
    出现时间不一致的问题。
    """
    return datetime.now(timezone.utc)


class IntentType(str, Enum):
    """
    客服系统支持的四种用户意图。
    """

    KNOWLEDGE_QUERY = "knowledge_query"
    SERVICE_REQUEST = "service_request"
    CHITCHAT = "chitchat"
    UNKNOWN = "unknown"


class ChatRequest(BaseModel):
    """
    用户发送给客服Agent的请求。
    """

    question: str = Field(
        min_length=1,
        max_length=1000,
        description="用户提出的问题",
    )

    conversation_id: str | None = Field(
        default=None,
        description="会话ID，第一版命令行程序中可以为空",
    )


class IntentResult(BaseModel):
    """
    意图识别模块的输出。
    """

    intent: IntentType

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="意图识别置信度",
    )

    reason: str = Field(
        min_length=1,
        description="模型为什么判断为这个意图",
    )

    fallback_used: bool = Field(
        default=False,
        description="是否因模型异常使用了安全降级结果",
    )

    error: str | None = Field(
        default=None,
        description="意图识别失败时的内部错误类型",
    )


class KnowledgeMatch(BaseModel):
    """
    知识库返回的一条匹配结果。
    """

    chunk_id: str = Field(
        min_length=1,
        description="知识片段的唯一编号",
    )

    source: str = Field(
        min_length=1,
        description="知识片段来自哪个文件",
    )

    content: str = Field(
        min_length=1,
        description="知识片段原文",
    )

    score: float = Field(
        ge=-1.0,
        le=1.0,
        description="关键词分和向量分融合后的检索分数",
    )

    vector_score: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="向量余弦相似度",
    )

    keyword_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="归一化后的BM25关键词分数",
    )

    rerank_score: float | None = Field(
        default=None,
        description="Cross-Encoder二阶段精排分数",
    )

    retrieval_method: str = Field(
        default="hybrid",
        description="产生当前结果的检索方式",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="页码、标题路径、URL、函数名和行号等来源信息",
    )


class ToolCallRecord(BaseModel):
    """
    一次工具调用的完整记录。
    """

    tool_name: str = Field(
        min_length=1,
        description="被调用的工具名称",
    )

    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="传给工具的参数",
    )

    success: bool = Field(
        description="工具是否调用成功",
    )

    result: dict[str, Any] | None = Field(
        default=None,
        description="工具成功时返回的数据",
    )

    error: str | None = Field(
        default=None,
        description="工具失败时的错误信息",
    )

    duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="工具执行耗时，单位为毫秒",
    )


class AgentResponse(BaseModel):
    """
    客服Agent最终返回给用户的结果。
    """

    request_id: str = Field(
        min_length=1,
        description="本次请求的唯一编号",
    )

    answer: str = Field(
        min_length=1,
        description="返回给用户的最终回答",
    )

    intent: IntentType = Field(
        description="识别出的用户意图",
    )

    sources: list[KnowledgeMatch] = Field(
        default_factory=list,
        description="回答使用的知识来源",
    )

    tool_calls: list[ToolCallRecord] = Field(
        default_factory=list,
        description="本次请求发生的工具调用",
    )

    need_human: bool = Field(
        default=False,
        description="是否需要转人工客服",
    )

    auto_resolved: bool = Field(
        default=False,
        description="问题是否由系统自动解决",
    )

    total_duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="整条Agent链路的总耗时",
    )


class EvaluationRecord(BaseModel):
    """
    保存到数据库中的评测记录。

    它记录一次请求经历了什么，
    方便后面分析Agent效果。
    """

    request_id: str = Field(
        min_length=1,
        description="本次请求的唯一编号",
    )

    question: str = Field(
        min_length=1,
        description="用户的原始问题",
    )

    predicted_intent: IntentType = Field(
        description="系统预测的意图",
    )

    intent_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="意图识别置信度",
    )

    intent_reason: str = Field(
        min_length=1,
        description="意图识别依据",
    )

    retrieved_chunk_ids: list[str] = Field(
        default_factory=list,
        description="检索到的知识片段编号",
    )

    retrieval_scores: list[float] = Field(
        default_factory=list,
        description="知识片段的混合检索分数",
    )

    tool_names: list[str] = Field(
        default_factory=list,
        description="调用过的工具名称",
    )

    answer: str = Field(
        min_length=1,
        description="Agent最终回答",
    )

    need_human: bool = Field(
        default=False,
        description="是否转人工",
    )

    success: bool = Field(
        default=True,
        description="整条技术链路是否成功完成",
    )

    auto_resolved: bool = Field(
        default=False,
        description="问题是否由系统自动解决",
    )

    error: str | None = Field(
        default=None,
        description="链路失败时的错误信息",
    )

    total_duration_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="整条链路耗时",
    )

    user_feedback: Literal[-1, 1] | None = Field(
        default=None,
        description="-1表示没帮助，1表示有帮助",
    )

    created_at: datetime = Field(
        default_factory=utc_now,
        description="记录创建时间",
    )
