# Autonomous Customer Support & Action Engine

**Enterprise-grade Agentic AI platform** — multi-agent orchestration (A2A),
hybrid RAG, CNN-based visual defect inspection, Model Context Protocol (MCP)
tool integration, semantic caching, human-in-the-loop governance, and full
observability — containerized for one-command deployment.

> ### Designed and Developed by **NIKHIL CHARY SRIRAMOJU**
> GitHub: [github.com/Nikhil-creat](https://github.com/Nikhil-creat)
> LinkedIn: [in.linkedin.com/in/nikhil-chary-sriramoju-95041b38a](https://in.linkedin.com/in/nikhil-chary-sriramoju-95041b38a)
> Email: sriramojunikhil66@gmail.com
> Instagram: [instagram.com/nikhil__sriramoju](https://www.instagram.com/nikhil__sriramoju)

---

## What makes this "agentic," not just "an LLM wrapper"

| Capability | Where |
|---|---|
| **A2A multi-agent delegation** via Agent Cards | `RouterSupervisorNode` |
| **Autonomous background agent** — monitors SLAs and self-escalates with *no user prompt* | `app/agents/proactive_agent.py` |
| **Self-correction loop** — captures tool tracebacks, repairs its own arguments, retries | `ActionExecutorNode` |
| **Self-verification** — a critic agent checks the executor's own output before responding | `VerificationCriticNode` |
| **Computer vision (CNN)** — classifies uploaded return/damage photos and feeds the verdict back into planning | `app/vision/` |
| **Hybrid RAG** — dense + BM25 sparse fusion, cross-encoder re-ranked | `app/tools/knowledge_base.py` |
| **MCP protocol** — dynamic tool discovery/invocation over JSON-RPC (HTTP + stdio) | `app/mcp/client.py` |
| **Semantic vector cache** — skips the whole graph on near-duplicate queries | `app/cache/semantic_cache.py` |
| **Human-in-the-loop governance** — `interrupt_before` pauses the graph for high-value/destructive actions, resumable from a Postgres checkpoint | `app/agents/graph.py` |
| **Dual-layer PII guardrails** — regex + spaCy NER, scrubbed before every LLM call and every persisted record | `app/security/pii_sanitizer.py` |
| **Real-time observability** — SSE + WebSocket trajectory streaming, OpenTelemetry + LangSmith tracing | `app/api/main.py`, `app/telemetry/` |
| **Automated evaluation** — DeepEval on Contextual Precision, Faithfulness, Tool Correctness, Toxicity | `eval/evaluate.py` |
| **Fully containerized** — API, Postgres, Redis, MCP tool server, OTEL collector, one `docker compose up` | `docker/` |

## Architecture

```
                         +----------------------------+
   User (chat + photo) ->|   FastAPI  (SSE / WS)       |
                         +--------------+-------------+
                                        v
                        +---------------------------------+
                        |   LangGraph StateGraph            |
                        |   (Postgres checkpointed)         |
                        +---------------+-------------------+
                                        v
                          RouterSupervisorNode (A2A)
                          +----------+----------+
                          v                     v
              VisualInspectionNode        PolicyRAGNode
               (CNN defect check)      (hybrid dense+BM25 RAG)
                          +----------+----------+
                                     v
                         ActionExecutorNode  <-> MCP tools
                       (self-correction retry loop)
                                     |
                     [HITL interrupt: high-value / destructive]
                                     v
                         VerificationCriticNode
                                     |
                            final_response

   -- running independently, in parallel --
   ProactiveSLAAgent: polls ApprovalQueue every N seconds,
   auto-escalates stale approvals with no human trigger.
```

## Tech stack

**Agentic orchestration:** LangGraph (StateGraph, `AsyncPostgresSaver`, `interrupt_before`)
**Protocols:** Model Context Protocol (MCP) over JSON-RPC (Streamable HTTP + stdio), A2A Agent Cards
**RAG:** BM25 sparse + dense embedding fusion, cross-encoder re-ranking
**Computer vision:** PyTorch CNN (residual-block architecture) for product defect/damage classification
**Backend:** Async FastAPI, SQLAlchemy 2.0 async ORM, asyncpg, Redis (semantic cache)
**Observability:** OpenTelemetry (OTLP), LangSmith
**Evaluation:** DeepEval (RAGAS-compatible metrics)
**Containerization:** Docker multi-stage builds, docker-compose (API + Postgres + Redis + MCP stub + OTEL collector)

## Running with Docker (recommended)

```bash
cd docker
OPENAI_API_KEY=sk-... docker compose up --build
```

This brings up:
- `api` — the FastAPI/LangGraph engine on `:8000`
- `postgres` — system-of-record + LangGraph checkpoints
- `redis` — semantic vector cache
- `mcp-tool-server` — a mock MCP tool server (swap for your real backend)
- `otel-collector` — trace collection endpoint on `:4317`

Check it's up:
```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/          # project + author info
```

## Running locally (without Docker)

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # optional, enables NER PII pass
uvicorn app.api.main:app --reload --port 8000
```

Stream a turn (text only):
```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"user_id": "u-1", "message": "Cancel order ORD-1029"}'
```

Stream a turn with a damage-claim photo (CNN inspection kicks in automatically):
```bash
curl -N -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d "{\"user_id\": \"u-1\", \"message\": \"My item arrived damaged, please refund me\", \"image_base64\": \"$(base64 -w0 damage_photo.jpg)\"}"
```

Run the evaluation suite:
```bash
python -m eval.evaluate
```

## Production hardening notes

- Swap `_fake_embed` in `app/api/main.py` for a real embeddings call.
- Swap `CrossEncoderReranker` for a hosted cross-encoder model.
- Train `ProductDefectCNN` on real labeled return photos and mount the
  weights at `CNN_WEIGHTS_PATH` (the container ships a `/app/models` volume
  for exactly this).
- Point `mcp_server_endpoints` at real MCP tool servers exposing
  `query_knowledge_base`, `execute_order_action`, `process_high_value_refund`,
  and `classify_product_defect_image` — the bundled `mcp-tool-server` is a
  mock for local development only.
- `SemanticCache`'s brute-force scan should be replaced with RediSearch's
  native HNSW vector index for large cache sizes.
- Replace `default_notify_sink` in `ProactiveSLAAgent` with a real
  Slack/PagerDuty/webhook integration.

---

*Autonomous Customer Support & Action Engine — Designed and Developed by
**Nikhil Chary Sriramoju**. Connect on
[GitHub](https://github.com/Nikhil-creat) ·
[LinkedIn](https://in.linkedin.com/in/nikhil-chary-sriramoju-95041b38a) ·
[Instagram](https://www.instagram.com/nikhil__sriramoju) ·
sriramojunikhil66@gmail.com*
