import json
import logging
import uuid

from app.agent import (
    create_customer_service_agent,
)
from app.schemas import AgentResponse, KnowledgeMatch


logger = logging.getLogger(__name__)


def format_source_location(source: KnowledgeMatch) -> str:
    """把不同文档类型的定位元数据转换成可读文本。"""
    metadata = source.metadata
    locations: list[str] = []

    if metadata.get("page"):
        locations.append(f"第{metadata['page']}页")

    title_path = metadata.get("title_path")

    if isinstance(title_path, list) and title_path:
        locations.append(
            "标题：" + " > ".join(str(item) for item in title_path)
        )

    if metadata.get("url"):
        locations.append(f"URL：{metadata['url']}")

    if metadata.get("symbol"):
        locations.append(f"符号：{metadata['symbol']}")

    start_line = metadata.get("start_line")
    end_line = metadata.get("end_line")

    if start_line and end_line:
        locations.append(f"第{start_line}-{end_line}行")

    return "；".join(locations)


def print_help() -> None:
    """
    打印命令行帮助信息。
    """
    print()
    print("可用命令：")
    print("  /help   查看帮助")
    print("  /stats  查看系统运行统计")
    print("  /good   评价上一条回答有帮助")
    print("  /bad    评价上一条回答没有帮助")
    print("  /exit   退出程序")
    print()


def print_response(
    response: AgentResponse,
) -> None:
    """
    用适合命令行阅读的格式展示Agent回答。
    """
    print()
    print("=" * 60)
    print(f"客服：{response.answer}")
    print(f"意图：{response.intent.value}")
    print(f"请求编号：{response.request_id}")
    print(
        f"总耗时：{response.total_duration_ms:.2f} ms"
    )

    if response.auto_resolved:
        print("处理状态：已由系统自动解决")
    elif response.need_human:
        print("处理状态：需要人工客服处理")
    else:
        print("处理状态：需要用户补充信息或稍后重试")

    if response.sources:
        print()
        print("知识来源：")

        for index, source in enumerate(
            response.sources,
            start=1,
        ):
            print(
                f"  [{index}] {source.source}"
                f"（融合分：{source.score:.4f}）"
            )
            print(
                "      分数明细："
                f"向量={source.vector_score}，"
                f"BM25={source.keyword_score:.4f}，"
                f"Rerank={source.rerank_score}"
            )
            print(
                f"      片段编号：{source.chunk_id}"
            )

            location = format_source_location(source)

            if location:
                print(f"      定位：{location}")

            print(
                f"      原文：{source.content}"
            )

    if response.tool_calls:
        print()
        print("工具调用：")

        for index, tool_call in enumerate(
            response.tool_calls,
            start=1,
        ):
            status = (
                "成功"
                if tool_call.success
                else "失败"
            )

            print(
                f"  [{index}] {tool_call.tool_name}"
            )
            print(f"      状态：{status}")
            print(
                f"      耗时："
                f"{tool_call.duration_ms:.2f} ms"
            )

            if tool_call.error:
                print(
                    f"      错误：{tool_call.error}"
                )

    print("=" * 60)
    print(
        "可以输入 /good 或 /bad "
        "评价这次回答。"
    )


def print_stats(summary: dict) -> None:
    """
    打印系统运行统计。
    """
    print()
    print("=" * 60)
    print("系统运行统计：")
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        )
    )
    print("=" * 60)


def main() -> None:
    """
    命令行程序入口。
    """
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(name)s "
            "%(message)s"
        ),
    )

    print("=" * 60)
    print("智能客服 Agent")
    print("=" * 60)
    print("正在加载模型和知识库，请稍候……")

    try:
        agent = create_customer_service_agent()

    except Exception as error:
        logger.exception("客服Agent启动失败")

        print()
        print(
            "启动失败："
            f"{type(error).__name__}: {error}"
        )
        print(
            "请检查.env、模型服务和知识库配置。"
        )
        return

    print("客服Agent启动成功。")
    print(
        "你可以咨询公司信息、客服时间和退款政策。"
    )

    print_help()

    # 为本次命令行运行生成会话编号
    conversation_id = (
        f"conversation_{uuid.uuid4().hex}"
    )

    # 保存上一条请求编号，用于提交用户评价
    last_request_id: str | None = None

    while True:
        try:
            user_input = input("你：").strip()

        except (KeyboardInterrupt, EOFError):
            print()
            print("已退出智能客服。")
            break

        if not user_input:
            continue

        command = user_input.lower()

        if command in {
            "/exit",
            "exit",
            "quit",
        }:
            print("已退出智能客服。")
            break

        if command == "/help":
            print_help()
            continue

        if command == "/stats":
            try:
                summary = (
                    agent
                    .evaluation_repository
                    .get_summary()
                )

                print_stats(summary)

            except Exception as error:
                logger.exception(
                    "查询统计数据失败"
                )

                print(
                    "统计数据查询失败："
                    f"{type(error).__name__}"
                )

            continue

        if command in {
            "/good",
            "/bad",
        }:
            if last_request_id is None:
                print(
                    "当前还没有可以评价的回答。"
                )
                continue

            feedback = (
                1
                if command == "/good"
                else -1
            )

            try:
                updated = (
                    agent
                    .evaluation_repository
                    .update_feedback(
                        request_id=last_request_id,
                        feedback=feedback,
                    )
                )

                if updated:
                    feedback_text = (
                        "有帮助"
                        if feedback == 1
                        else "没有帮助"
                    )

                    print(
                        f"评价已保存：{feedback_text}"
                    )
                else:
                    print(
                        "没有找到对应的评测记录。"
                    )

            except Exception as error:
                logger.exception(
                    "保存用户评价失败"
                )

                print(
                    "评价保存失败："
                    f"{type(error).__name__}"
                )

            continue

        try:
            response = agent.run(
                question=user_input,
                conversation_id=conversation_id,
            )

            last_request_id = (
                response.request_id
            )

            print_response(response)

        except Exception as error:
            logger.exception(
                "处理用户问题失败"
            )

            print()
            print(
                "客服系统处理失败："
                f"{type(error).__name__}: {error}"
            )
            print(
                "请检查输入内容或稍后重试。"
            )
