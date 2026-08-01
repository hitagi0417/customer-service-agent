# 智能客服 Agent

这个项目面向企业客服场景，解决人工查询知识库和重复回答效率低的问题。核心链路为：

`会话记忆 → 问题改写 → 意图识别 → 受控规划 → 业务工具 → 引用校验 → 评测记录`

## 多轮记忆与受控 Agent

每个会话通过 `conversation_id` 持久化到 SQLite，并与 `customer_id` 绑定，避免跨客户读取历史或订单。上下文由“结构化滚动摘要 + 最近消息窗口”组成；当前追问会先改写成可独立理解的问题，再进入意图识别和检索，避免直接把无限历史塞进 Prompt。

知识咨询和业务请求使用最多 `AGENT_MAX_STEPS` 步的 Planner。Planner 每一步只能选择以下白名单动作：

- `search_knowledge_base`：检索知识库；最终回答必须引用真实返回的 `chunk_id`。
- `query_order`：查询当前客户订单。
- `check_refund_eligibility`：根据订单状态、签收时间和退款窗口做只读判断。
- `create_service_ticket`：证据不足、工具失败或确需人工时创建幂等工单。
- `ask_clarification` / `finish`：追问缺失信息或结束执行。

订单工具的 `customer_id`、工单的 `request_id` 都由服务端注入，模型输出无法覆盖。示例订单为 `DEMO-1001`（已签收）和 `DEMO-1002`（已发货）。API 响应与评测记录会保存问题改写结果、规划轨迹、各阶段耗时和累计 Token 用量，便于复盘成本与 Bad Case。

当前 `customer_id` 是便于本地演示的数据隔离参数，并不替代真实认证；生产环境应从登录态或 JWT 声明中解析客户身份，不能直接信任请求体中的身份字段。

## 多格式知识库

把文件放进 `knowledge/` 后，启动项目会自动递归扫描并解析。

项目自带一组可复现的客服知识样例，共18个文件；版本过滤后有15个有效文件、22个知识片段，覆盖公司信息、客服时间、退款、换货、物流、相似产品型号、保修、支付发票、账户安全、会员积分和订单FAQ。不同政策被拆成独立文档，方便测试相似内容之间的召回和排序。

| 类型 | 扩展名 | 解析方式 | 保留的定位信息 |
| --- | --- | --- | --- |
| 文本 | `.txt` | 整篇读取后按自然边界切片 | 文件名 |
| Markdown | `.md`、`.markdown` | 按标题层级拆分 | 标题路径、Front Matter |
| PDF | `.pdf` | 按页提取文本层 | 页码、总页数 |
| HTML | `.html`、`.htm` | 删除脚本和导航等噪声后提取正文 | 页面标题 |
| Python | `.py` | 使用 AST 按顶层类和函数拆分 | 符号名、类型、行号 |
| 其他代码 | `.js`、`.ts`、`.java`、`.go` 等 | 按重叠行窗口拆分 | 语言、起止行号 |

PDF 当前支持带文本层的文件。扫描版 PDF 需要先做 OCR，否则无法提取正文。

## 脏数据处理

知识入库时会执行以下质量控制：

- 每个文件独立解析；损坏文件会被隔离，不影响其他知识源。
- 空内容、过短内容和不包含有效文字或数字的内容会被过滤。
- 使用文件哈希过滤重复文件，使用内容哈希过滤重复片段。
- PDF页面没有文本但包含图片时，会标记为 `ocr_required`。
- 每次构建都会生成 `data/ingestion_report.json`，记录失败、过滤、重复和待OCR内容。
- Markdown中的 `status`、`version` 和 `effective_date` 会真正参与入库决策：停用、草稿和未来生效文档不进入索引，同一 `doc_id` 只选择当前最高有效版本。

最小有效内容长度可以通过 `.env` 调整：

```dotenv
KNOWLEDGE_MIN_CONTENT_CHARS=10
INGESTION_REPORT_PATH=data/ingestion_report.json
```

### Markdown Front Matter 示例

```markdown
---
doc_id: refund-policy
department: customer-service
---

# 退款政策

购买后七天内可以申请退款。
```

`doc_id` 和 `department` 会进入每个知识片段的元数据，便于后续做更新、删除和过滤。

### 受控网页知识源

本地 HTML 文件不需要额外配置。只有从 URL 抓取网页时，才执行下面的步骤：

1. 将 `knowledge/web_sources.example.json` 复制为 `knowledge/web_sources.json`。
2. 在 JSON 文件中填写网页 URL。
3. 在 `.env` 中配置：

```dotenv
ENABLE_WEB_INGESTION=true
WEB_SOURCES_PATH=knowledge/web_sources.json
WEB_ALLOWED_DOMAINS=docs.example.com
```

网页抓取只接受 HTTPS、只允许精确命中白名单域名、拒绝重定向，并限制响应类型和大小。

## 安装与运行

```powershell
python -m pip install -r requirements-dev.txt
python run.py
```

首次使用时，先参考 `.env.example` 补全 `.env` 中的模型配置。

## 测试与评测

```powershell
python -m pytest -q
python tests/evaluate.py
python tests/evaluate.py --cases tests/eval_holdout_cases.json --output data/eval_holdout_report.json
python tests/evaluate_multiturn.py
```

单元测试覆盖各格式解析、知识加载、意图识别、会话隔离、滚动摘要、受控规划、订单权限边界和工具调用。自动评测会执行完整 Agent 链路，并将结果写入 `data/eval_report.json`。

评测分为54题开发回归集、12题独立 holdout 集和4组多轮对话集。指标包含意图准确率、来源 Recall@K、安全拒答、工具选择准确率、转人工率、分阶段耗时和平均 Token。多轮集额外检查指代补全、缺失槽位追问、上下文订单号继承与 Prompt Injection。关键词概念匹配只用于稳定回归，不等同于人工质量评分；对外表述指标时应同时注明评测集、模型、日期和评测方法。

## 混合检索

检索层同时使用向量相似度和BM25关键词分：

```text
用户问题
  ├─ 向量召回：处理同义表达和语义改写
  └─ BM25召回：处理产品型号、订单号和政策名称等精确词
          ↓
      候选集合并
          ↓
      加权分数融合
          ↓
      Top10候选
          ↓
 Cross-Encoder Rerank
          ↓
        Top3
```

每条检索结果会同时保留 `vector_score`、`keyword_score`、融合 `score` 和 `rerank_score`，便于评测与Bad Case分析。Rerank默认使用支持中英文的 `BAAI/bge-reranker-base`，可以通过环境变量关闭。

可以使用同一套标准来源问题比较三种检索模式：

```powershell
python tests/evaluate_retrieval.py
```

报告会保存到 `data/retrieval_comparison.json`，包含纯向量、纯BM25、混合检索和混合检索加Rerank的 Recall@1、Recall@3、MRR、平均耗时、P95耗时以及每道题的排序明细。

## 关键目录

```text
app/
  parsers/          # 文档解析器与解析器注册表
  knowledge.py      # 统一加载、清洗、切片和元数据生成
  bm25.py           # 中文关键词切词和BM25索引
  retrieval.py      # BM25与向量混合检索
  conversation.py   # 会话持久化、近期窗口与滚动摘要
  planning.py       # 结构化 Planner 动作协议
  agent.py          # 多轮改写、受控执行与引用校验
knowledge/          # 本地知识文件和网页源清单
tests/              # 单元测试、开发集、holdout与多轮评测
```
