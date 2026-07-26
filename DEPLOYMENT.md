# 智能客服 Agent 部署指南

项目采用“一容器一进程”的部署方式。Embedding 和 Rerank 模型只在应用启动时加载一次；SQLite、评测记录和工单数据保存在 Docker 数据卷中。

## 一、部署前准备

建议服务器至少满足：

- Linux x86_64；
- 4 核 CPU；
- 8 GB 内存；
- 20 GB 可用磁盘；
- 已安装 Docker Engine 和 Docker Compose；
- 安全组只开放 `22`、`80`、`443`，不要直接向公网开放 `8000`。

首次启动需要从 Hugging Face 下载 Embedding 和 Rerank 模型，所以耗时会明显长于后续启动。

## 二、配置生产环境

先复制环境变量模板：

```bash
cp .env.example .env
```

生成 API 访问密钥：

```bash
openssl rand -hex 32
```

编辑 `.env`，至少正确填写：

```dotenv
LLM_API_KEY=大模型服务的密钥
LLM_BASE_URL=大模型服务地址
LLM_MODEL_NAME=模型名称

SERVICE_API_KEY=刚才生成的64位随机字符串
API_MAX_CONCURRENCY=1
API_REQUEST_TIMEOUT_SECONDS=60
```

`SERVICE_API_KEY` 是调用本项目 HTTP API 的密钥，不是大模型密钥。生产模式如果缺少它，服务会拒绝启动。

## 三、启动 Docker 服务

在项目根目录执行：

```bash
docker compose up -d --build
```

查看启动日志：

```bash
docker compose logs -f customer-service-agent
```

看到服务启动完成后，检查健康状态：

```bash
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
```

预期分别返回：

```json
{"status":"alive"}
{"status":"ready"}
```

## 四、调用客服接口

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: 你的SERVICE_API_KEY" \
  -d '{"question":"商品签收后几天内可以退货？","conversation_id":"demo-001"}'
```

查看本进程的基础运行指标：

```bash
curl http://127.0.0.1:8000/metrics \
  -H "X-API-Key: 你的SERVICE_API_KEY"
```

指标包括总请求数、成功数、失败数、超时数、平均耗时和运行时间。Agent 自身仍会把意图、召回片段、工具调用、答案和耗时记录到 SQLite 评测表中。

## 五、配置域名与 HTTPS

生产环境应在容器前放置 Nginx 或 Caddy，由反向代理负责：

- 绑定域名；
- 自动签发和续期 HTTPS 证书；
- 限制请求体大小；
- 设置访问日志和限流；
- 把请求转发到 `127.0.0.1:8000`。

最小 Nginx 转发配置：

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

正式上线时再用 Certbot 或 Caddy 配置 HTTPS，不要把 `.env`、SQLite 文件或模型密钥提交到代码仓库。

## 六、更新与回滚

更新知识库或代码后：

```bash
docker compose up -d --build
docker compose logs --tail=100 customer-service-agent
```

查看容器状态：

```bash
docker compose ps
```

停止服务但保留数据：

```bash
docker compose down
```

不要随意执行 `docker compose down -v`，因为 `-v` 会删除工单、评测数据和模型缓存的数据卷。

## 七、当前容量边界

当前配置每个容器同时只执行 1 个 Agent 请求，其他请求会等待，目的是避免 CPU 模型互相争抢资源以及 SQLite 写入冲突。它适合个人演示和小流量校招项目，但不能据此宣称已经承载 100 个并发用户。

要提高容量，应先用压测得到单实例吞吐和 P95，再横向增加容器副本。多副本部署前，需要把 SQLite 替换为 PostgreSQL，并让 Nginx 或云负载均衡器分发请求。不要简单增加 Uvicorn worker，因为每个 worker 都会各自加载一份模型，占用数倍内存。
