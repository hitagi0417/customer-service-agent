# 智能客服 Agent

这个项目面向企业客服场景，解决人工查询知识库和重复回答效率低的问题。核心链路为：

`用户问题 → 意图识别 → 知识检索/业务工具 → 受约束回答 → 评测记录`

项目提供两套运行形态：

- 本地开发/单元测试：SQLite + 内存向量索引，不依赖中间件。
- Docker生产形态：PostgreSQL + Qdrant + Redis，支持共享数据、向量持久化和检索缓存。

```text
FastAPI / Agent
  ├─ PostgreSQL：工单、执行记录、评测数据
  ├─ Qdrant：知识向量、增量更新、删除同步
  └─ Redis：检索结果缓存（知识版本变化后自动换Key失效）
```

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

需要启动完整生产依赖时：

```powershell
docker compose up -d --build
```

完整步骤见 `DEPLOYMENT.md`。

## 测试与评测

```powershell
python -m pytest -q
python tests/evaluate.py
```

当前共51个单元测试，覆盖各格式解析、知识加载、意图识别、检索、工具调用、Token统计、反馈接口、可观测性持久化、压测报告计算和Qdrant增量同步。自动评测会执行完整 Agent 链路，并将结果写入 `data/eval_report.json`。

当前端到端评测集包含54题，并按直接检索、精确关键词、语义改写、政策边界、多片段信息、相似型号干扰、版本冲突、HTML来源和无答案拒答等类型统计准确率。关键词评测支持同义概念组，例如“不能修改/无法修改/不可以修改”任意一个命中均可，避免把正确的自然语言改写误判为错误。

2026-07-28使用当前代码和固定评测集得到的基线如下，原始逐题结果见`data/eval_report.json`：

| 指标 | 结果 |
| --- | ---: |
| 评测题通过数 | 54/54 |
| Recall@TopK / 回答准确率 | 100% / 100% |
| 引用准确率 / 无答案拒答准确率 | 100% / 100% |
| 工具调用成功率 | 100%（51/51） |
| 自动解决率 / 人工接管率 | 75.93% / 16.67% |
| 平均耗时 / P95耗时 | 2146.64ms / 3100.81ms |
| 总Token / 单请求平均Token | 66720 / 1235.56 |

这是项目自建的54题回归集结果，用于比较同一数据集上的代码和参数变更，不代表未知线上问题也能达到100%。下一阶段应持续收集真实用户反馈和Bad Case扩充评测集。

只复现某一道Bad Case时，不必重新跑完整评测：

```powershell
python tests/evaluate.py --case-id knowledge_042
```

### 评测指标

离线评测报告包含：

- `retrieval_recall_at_k`：应命中的知识来源是否出现在TopK召回结果中；
- `answer_accuracy`：回答是否覆盖评测集定义的必需事实或同义表达；
- `citation_precision`、`citation_source_recall`：回答引用的来源是否正确、必需来源是否被引用；
- `knowledge_abstention_accuracy`：知识库没有答案时是否拒答并安全转人工；
- `tool_success_rate`：检索和工单工具的真实执行成功率；
- `pipeline_success_rate`、`auto_resolution_rate`、`human_transfer_rate`：完整链路成功、自动解决和人工接管情况；
- `average_duration_ms`、`p95_duration_ms`：平均与P95响应耗时；
- `total_tokens`、`average_tokens_per_request`、`estimated_cost_usd`：Token消耗和估算成本。

`answer_accuracy`是基于人工编写标准题和必需事实的可复现规则评测，不等同于开放域语义评分。成本只有在`.env`填写供应商当前单价后才有意义：

```dotenv
LLM_INPUT_COST_PER_1M_TOKENS=输入每百万Token美元单价
LLM_OUTPUT_COST_PER_1M_TOKENS=输出每百万Token美元单价
```

若OpenAI兼容供应商不返回`usage`，系统会记录模型调用但不会伪造Token数；这时可通过`token_usage_coverage_rate`看到数据覆盖率。

### 线上反馈与指标

每次`/api/chat`返回唯一的`request_id`。用户可对该次回答提交有帮助或无帮助反馈：

```powershell
$headers = @{
  "X-API-Key" = $env:SERVICE_API_KEY
  "Content-Type" = "application/json"
}

Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/api/feedback" `
  -Headers $headers `
  -Body '{"request_id":"返回的request_id","feedback":1}'
```

`GET /metrics`同时返回当前进程指标和数据库累计指标，包括工具成功率、引用ID有效率、Token覆盖率、成本、反馈率、自动解决率和人工接管率。完整工具执行与引用会分别写入`tool_execution_records`和`citation_records`，便于定位失败发生在哪一层。

### 并发压测

对已经启动的真实API执行压测：

```powershell
$env:SERVICE_API_KEY = "你的服务API Key"
python tests/load_test.py `
  --base-url http://127.0.0.1:8000 `
  --concurrency 100 `
  --requests 200 `
  --confirm-real-llm-cost
```

报告保存到`data/load_test_report.json`，包含成功率、吞吐、P50/P95/P99、HTTP状态码和错误分布。压测会真实调用模型并产生费用，因此脚本要求显式确认；没有目标服务器、模型限额和机器配置时，项目不会宣称“已扛住100并发”。

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

生产模式下，向量存入Qdrant。启动或热更新知识库时，系统用稳定的`chunk_id + content_hash + embedding_model`判断差异，只编码新增/变更片段，并删除知识库中已经不存在的向量。Redis缓存Key包含完整知识指纹，因此文档更新后旧缓存不会被继续命中。

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
  vector_store.py   # Qdrant增量同步和向量召回
  cache.py          # Redis检索缓存及故障降级
  database.py       # SQLAlchemy表结构与连接池
  telemetry.py      # Token usage兼容提取、累计和成本估算
  evaluation.py     # 链路记录、反馈与线上聚合指标
  tickets.py        # 跨SQLite/PostgreSQL的工单仓储
  agent.py          # Agent 主链路与受约束回答
migrations/         # Alembic数据库版本迁移
knowledge/          # 本地知识文件和网页源清单
tests/              # 单元测试、54题端到端评测、检索对比和压测
```
