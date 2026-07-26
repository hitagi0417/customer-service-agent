# 智能客服 Agent

这个项目面向企业客服场景，解决人工查询知识库和重复回答效率低的问题。核心链路为：

`用户问题 → 意图识别 → 知识检索/业务工具 → 受约束回答 → 评测记录`

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
```

单元测试覆盖各格式解析、知识加载、意图识别、检索和工具调用。自动评测会执行完整 Agent 链路，并将结果写入 `data/eval_report.json`。

当前端到端评测集包含54题，并按直接检索、精确关键词、语义改写、政策边界、多片段信息、相似型号干扰、版本冲突、HTML来源和无答案拒答等类型统计准确率。关键词评测支持同义概念组，例如“不能修改/无法修改/不可以修改”任意一个命中均可，避免把正确的自然语言改写误判为错误。

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
  agent.py          # Agent 主链路与受约束回答
knowledge/          # 本地知识文件和网页源清单
tests/              # 单元测试与20题端到端评测
```
