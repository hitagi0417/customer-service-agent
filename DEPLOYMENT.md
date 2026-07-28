# 智能客服 Agent 部署指南

生产形态由五个组件组成：

```text
Nginx / Caddy
      ↓
FastAPI Agent
  ├─ PostgreSQL：工单与评测数据
  ├─ Qdrant：持久化知识向量
  ├─ Redis：检索结果缓存
  └─ 大模型API：意图识别与受约束回答
```

SQLite和内存向量索引只用于本地开发与测试，不再承担多实例生产数据。

## 一、部署前准备

建议演示服务器至少满足：

- Linux x86_64；
- 4核CPU、8GB内存、20GB可用磁盘；
- Docker Engine与Docker Compose；
- 安全组只开放`22`、`80`和`443`。

Compose中的`8000`、`5432`、`6333`、`6379`都只绑定`127.0.0.1`，不会直接暴露到公网。

首次启动需要下载Embedding和Rerank模型，耗时会明显长于后续启动。

## 二、配置生产环境

复制模板：

```bash
cp .env.example .env
```

生成只包含URL安全字符的随机密钥：

```bash
openssl rand -hex 32
```

至少填写：

```dotenv
LLM_API_KEY=大模型服务密钥
LLM_BASE_URL=大模型服务地址
LLM_MODEL_NAME=模型名称

SERVICE_API_KEY=一段独立的64位随机字符串
POSTGRES_PASSWORD=另一段独立的64位随机字符串
```

不要让`SERVICE_API_KEY`、`POSTGRES_PASSWORD`和大模型密钥复用。`.env`已被Git忽略，不能提交到仓库。

容器内会自动使用：

```dotenv
DATABASE_URL=postgresql+psycopg://agent:密码@postgres:5432/customer_service
VECTOR_STORE_BACKEND=qdrant
QDRANT_URL=http://qdrant:6333
ENABLE_RETRIEVAL_CACHE=true
REDIS_URL=redis://redis:6379/0
```

## 三、启动与检查

构建并启动完整服务：

```bash
docker compose up -d --build
docker compose ps
```

应用容器启动前会先执行`alembic upgrade head`，数据库结构变化因此有明确版本，而不是靠运行时手工改表。

查看应用日志：

```bash
docker compose logs -f customer-service-agent
```

首次入库日志会显示Qdrant新增的片段数。知识未变化时再次启动，新增或更新数应为`0`。

检查存活和就绪状态：

```bash
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
```

`live`只说明进程还活着；`ready`还会确认Agent已经加载、PostgreSQL可查询、Qdrant集合可访问。

## 四、调用接口

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: 你的SERVICE_API_KEY" \
  -d '{"question":"商品签收后几天内可以退货？","conversation_id":"demo-001"}'
```

查看进程指标：

```bash
curl http://127.0.0.1:8000/metrics \
  -H "X-API-Key: 你的SERVICE_API_KEY"
```

Agent会把意图、召回片段、工具调用、答案、耗时和用户反馈写入PostgreSQL。Redis故障时检索会自动绕过缓存，缓存故障不会伪装成业务执行成功。

## 五、知识更新与删除

修改`knowledge/`后重新构建应用：

```bash
docker compose up -d --build customer-service-agent
docker compose logs --tail=100 customer-service-agent
```

同步规则：

1. 新`chunk_id`：生成Embedding并写入Qdrant；
2. `content_hash`或Embedding模型变化：重新编码并覆盖；
3. 文档删除导致`chunk_id`消失：删除对应Qdrant点；
4. 知识集合变化：Redis缓存Key中的知识指纹变化，旧缓存自然失效。

如果更换Embedding模型后向量维度发生变化，应设置新的`QDRANT_COLLECTION`，确认新集合评测通过后再删除旧集合，避免误删可回滚数据。

## 六、域名与HTTPS

生产环境应在容器前放置Nginx或Caddy，负责HTTPS、访问日志、请求大小限制和限流。最小Nginx转发示例：

```nginx
server {
    listen 80;
    server_name agent.example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 70s;
    }
}
```

正式上线必须配置HTTPS，不应把8000端口直接暴露到公网。

## 七、备份、更新与回滚

停止服务但保留数据：

```bash
docker compose down
```

不要随意执行`docker compose down -v`，`-v`会删除PostgreSQL、Qdrant、Redis和模型缓存的数据卷。

正式部署应定期备份：

- PostgreSQL：使用`pg_dump`；
- Qdrant：使用集合Snapshot；
- 知识原文件：由Git或对象存储保留版本。

更新前先记录镜像版本和评测结果。出现Bad Case时回滚应用镜像、知识提交和Qdrant集合，而不是直接修改线上数据。

## 八、100人同时访问的容量边界

这套架构已经消除了SQLite单文件写入和内存向量无法共享的问题，可以让多个Agent实例共享PostgreSQL、Qdrant与Redis。但“100个连接能进入系统”不等于“100个Agent任务能同时推理”。

真正上线前仍需用压测获得单实例吞吐、P95、错误率和资源占用，再决定：

- `API_MAX_CONCURRENCY`；
- Agent容器副本数；
- PostgreSQL连接池大小；
- 大模型API限额；
- 是否把Embedding/Rerank迁移到GPU推理服务。

不要简单增加Uvicorn worker：每个worker都会单独加载Embedding和Rerank模型。更合理的扩容方式是用多个容器副本配合负载均衡，并把并发上限建立在压测数据上。
