# Deployment Guide for RAG Application

This guide explains how to deploy the RAG application using Docker and GitHub Container Registry (GHCR).

## Prerequisites

- GitHub account with permissions to the repository
- Docker and Docker Compose installed on your deployment machine
- Access to GitHub Container Registry (GHCR)
- Ollama running on the host machine for LLM functionality

## CI/CD Pipeline

The application uses GitHub Actions for continuous integration and delivery:

- Images are automatically built and published to GHCR when pushing to the `dev-ops` branch
- Tagged versions (e.g., `v1.0.0`) also trigger the build and publish workflow
- Pull requests to the `dev-ops` branch will build but not publish images

## Managing the Dev-Ops Branch

To set up and use the dev-ops branch for CI/CD:

```bash
# Create and switch to the dev-ops branch
git checkout -b dev-ops

# After making changes, commit them
git add .
git commit -m "Your commit message"

# Push to the remote repository
git push -u origin dev-ops
```

To create a release version:

```bash
# Create a tag
git tag -a v1.0.0 -m "First production release"

# Push the tag
git push origin v1.0.0
```

## Deploying from GHCR (Production)

### 1. Log in to GitHub Container Registry

```bash
# Login to GHCR
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
```

Replace `USERNAME` with your GitHub username and ensure you have a `GITHUB_TOKEN` environment variable set with a token that has `read:packages` scope.

### 2. Create a .env file (optional)

Create a `.env` file with custom settings:

```
REGISTRY=ghcr.io
GITHUB_REPOSITORY=yourusername/rag-app
TAG=latest
FRONTEND_PORT=3000
BACKEND_PORT=5000
OLLAMA_HOST_URL=http://host.docker.internal:11434
LLM_MODEL=gemma3:1b
```

### 3. Pull and run with Docker Compose

```bash
# Pull and start the application
docker-compose -f docker-compose.prod.yml pull
docker-compose -f docker-compose.prod.yml up -d
```

### 4. Setup Ollama (if not already running)

The application requires Ollama running on the host machine. To install:

1. Download Ollama from https://ollama.com/download
2. Install and start Ollama
3. Pull the required model:

```bash
ollama pull gemma3:1b
```

If you want to use a different model, specify it in the `.env` file or edit the `docker-compose.prod.yml` file.

## Local Development Deployment

For local development, use the standard docker-compose file:

```bash
docker-compose build
docker-compose up -d
```

## Accessing the Application

- Frontend: http://localhost:3000 (or your custom FRONTEND_PORT)
- Backend API: http://localhost:5000 (or your custom BACKEND_PORT)

## Benchmarking performance (Phase 4)

The evaluation harness (`python -m evaluation.run_benchmark`) can run up to three LLM-as-Judge calls in parallel per question/config when `EVAL_MAX_CONCURRENCY` is greater than `1` (see `python/config/models.py`). Client-side parallelism only helps if the Ollama server can serve that many concurrent generations: set **`OLLAMA_NUM_PARALLEL`** on the Ollama host to at least the same value (how you set it depends on install — e.g. systemd unit `Environment=OLLAMA_NUM_PARALLEL=3` or the equivalent for your platform). If the server stays at parallel `1`, extra client threads mostly queue at Ollama and wall-clock stays near serial.

Raising parallelism increases peak RAM: a rough rule for **gemma3:4b** at Q4_K_M is on the order of **~2.5 GiB × concurrency**; on an 8 GiB machine, `EVAL_MAX_CONCURRENCY=3` may be tight if other services are loaded. Start with `2` and watch `ollama` / system memory before going higher.

## Ollama tuning (Tier 3.7)

Two layers: **host** environment variables read by `ollama serve`, and **per-request** JSON fields this app sends on `/api/chat` and `/api/embed`.

### Host-side (Ollama server process)

Set these on the machine where Ollama runs (not in the app container unless you containerize Ollama). Examples:

**systemd** (`/etc/systemd/system/ollama.service` drop-in or override):

```ini
[Service]
Environment="OLLAMA_NUM_PARALLEL=2"
Environment="OLLAMA_FLASH_ATTENTION=1"
```

**macOS (launchd)** — add to the plist `EnvironmentVariables` or export before `ollama serve` in your shell profile.

| Variable | Effect |
|---|---|
| `OLLAMA_NUM_PARALLEL` | Concurrent model executions the server will schedule (raise only if RAM allows; pair with `EVAL_MAX_CONCURRENCY` for judge parallelism). |
| `OLLAMA_FLASH_ATTENTION` | Enables flash attention where supported (often a safe throughput win on GPU paths). |

**KV cache / gemma3:** do **not** set `OLLAMA_KV_CACHE_TYPE` to a non-default value for **gemma3** on Ollama **before 0.12.5** — known kernel fallback / quality regressions (upstream [#9683](https://github.com/ollama/ollama/issues/9683), [#10945](https://github.com/ollama/ollama/issues/10945), [#11949](https://github.com/ollama/ollama/issues/11949)). Prefer upgrading Ollama before experimenting with KV-cache quantization.

### Request-side (this repo — Python → Ollama)

Configured in [python/config/models.py](python/config/models.py) and forwarded by [python/llm/ollama_llm.py](python/llm/ollama_llm.py) (chat) and [python/embeddings/ollama_embed.py](python/embeddings/ollama_embed.py) (embeddings).

| Variable | Default | Effect |
|---|---|---|
| `OLLAMA_NUM_BATCH` | `512` | Passed as `options.num_batch` on `/api/chat` (batch size for prompt processing). |
| `OLLAMA_NUM_CTX` | `0` (unset) | When **> 0**, passed as `options.num_ctx`. **`0` means “use the model default”** — recommended so long prompts are not silently truncated. |
| `OLLAMA_KEEP_ALIVE` | `5m` | Top-level `keep_alive` on `/api/chat` and embedding requests; keeps models loaded between calls (raises RAM if set very high). |

Optional examples for `docker-compose.yml` (backend `environment:`) are commented next to `OLLAMA_HOST_URL`.

### Micro-benchmark CLI

Validate host + request knobs with before/after numbers (run from `python/` with Ollama reachable at `OLLAMA_HOST_URL`):

```bash
cd python
python -m evaluation.bench_ollama gen --model gemma3:4b --prompts 20 --concurrency 2
python -m evaluation.bench_ollama embed --model nomic-embed-text-v2-moe --items 200 --batch 32
python -m evaluation.bench_ollama gen --model gemma3:4b --prompts 10 --json   # machine-readable summary
```

Example **human-readable** excerpt (values depend on hardware; sequential vs concurrent shows queueing when `OLLAMA_NUM_PARALLEL` is low):

```
bench_ollama gen  model=gemma3:4b  requests=20  concurrency=2
  sequential: wall=45.20s  p50=2.100s  p95=2.800s  median tok/s=42.5
  concurrent: wall=28.10s  p50=2.050s  p95=3.400s  median tok/s=41.2
```

After raising `OLLAMA_NUM_PARALLEL` and enabling flash attention on the host, re-run the same command: **wall** for the concurrent phase and **median tok/s** should move; if parallelism is still `1`, concurrent wall time stays near sequential.

## Reranker defaults and RAM (Phase 3.6a)

Cross-encoder reranking is now enabled by default in dev compose (`SKIP_RERANKING=false`). This improves retrieval precision, but adds model-load latency and memory pressure.

- Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (smaller; recommended for 8GB hosts).
- Optional heavier model: `BAAI/bge-reranker-v2-m3` via `RERANKER_MODEL`, only if you have RAM headroom.
- Candidate pool before rerank is controlled by `RERANK_TOP_M` (default `50`, capped at `150`); increasing it can improve quality but raises latency.

If the host is memory-constrained, set `SKIP_RERANKING=true` to disable reranking, or keep reranking on and reduce `RERANK_TOP_M`.

## Troubleshooting

### Connection Issues

If the backend cannot connect to Qdrant or MongoDB:
- Check that all containers are running: `docker ps`
- Inspect the logs: `docker logs rag-app-backend-1`
- Ensure proper network connectivity: `docker network inspect rag-app_rag-network`

### PDF Upload Failures

If PDF uploads fail:
- Check if environment variables are being passed correctly
- Verify Qdrant is running and accessible
- Look for errors in the logs: `docker logs rag-app-backend-1`

## Updating the Application

To update to a new version:

```bash
# Pull the latest images
docker-compose -f docker-compose.prod.yml pull

# Restart the services
docker-compose -f docker-compose.prod.yml up -d
``` 