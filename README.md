# KnowledgeBuilder: Contextual Retrieval + Structured Data Docs

This project implements a modular indexing system for Spark/Flink repositories with:

1. Evidence-based structured markdown generation.
2. Contextual Retrieval (Anthropic pattern): contextualized chunks + hybrid retrieval.
3. Backend adapters for:
- AWS Bedrock Knowledge Base (`aws_kb`)
- Local FAISS (`faiss`)

## What is implemented

### 1) Structured knowledge generation

For each repository:
- Artifact docs for changed files.
- Table docs (`entities/tables/...`) combining:
  - Code evidence (read/write refs)
  - Glue Catalog schema/location when available
- Metric docs (`entities/metrics/...`) from detected formulas.
- Architecture overview (`entities/architecture/overview.md`).

Each entity doc includes frontmatter-like metadata:
- `entity_type`, `entity_id`, `repo`, `commit_sha`
- `source_of_truth` (`glue_catalog` or `code_inference`)
- `confidence` level
- explicit evidence file paths

Core code:
- `kb_indexer/extractor.py`
- `scripts/reindex.py`

### 2) Contextual Retrieval mechanism

Implemented in `kb_indexer/contextual_retrieval.py` and `kb_indexer/backends/faiss_local.py`:

- Split documents into semantic chunks.
- Generate chunk context:
  - `heuristic` mode
  - `external_command` mode (hook for Claude SDK / Strands)
- Build contextual chunk text:
  - `[Context] ... [Chunk] ...`
- Local retrieval uses:
  - dense retrieval (FAISS over contextual embeddings)
  - sparse retrieval (BM25 over contextual chunk text)
  - reciprocal rank fusion (RRF)
  - optional rerank step (query overlap feature)

This follows the contextual embedding + contextual BM25 + fusion strategy from Anthropic’s Contextual Retrieval article.

### 3) Multi-repo incremental indexing

- Configure multiple repositories in `kb_config.yaml`.
- Per-repo commit SHA state stored in:
  - `s3` (recommended for cron/fresh clones), or
  - local file.
- On each run:
  - compute git diff from last indexed SHA
  - update changed artifact docs
  - delete removed docs
  - rebuild entity docs per changed repo

## Setup

```bash
cp kb_config.example.yaml kb_config.yaml
uv sync
```

Optional extras:

```bash
# Local FAISS index + BM25
uv sync --extra faiss

# Better local embeddings
uv sync --extra local-embeddings
```

## Configure repositories and backend

Edit `kb_config.yaml`:

- `repositories`: list of repos to index.
- `backend.type`: `aws_kb` or `faiss`.
- `state.backend`: `s3` or `local`.
- `indexing.contextual_*`: contextual retrieval settings.

### Context generation modes

1. `indexing.contextualizer_mode: heuristic`
- no LLM call, deterministic context synthesis.

2. `indexing.contextualizer_mode: external_command`
- your command receives JSON input via stdin and returns:
```json
{"context":"..."}
```
- example stub: `scripts/contextualize_chunk_stub.py`

This is the hook for Claude Agent SDK or Strands-based context generation.

## Reindex

```bash
uv run python scripts/reindex.py --config kb_config.yaml --print-plan
```

Force full:

```bash
uv run python scripts/reindex.py --config kb_config.yaml --full --print-plan
```

Subset:

```bash
uv run python scripts/reindex.py --config kb_config.yaml --repos spark-jobs flink-jobs
```

## Query

Local FAISS (hybrid contextual retrieval):

```bash
uv run python scripts/search.py --config kb_config.yaml --query "which jobs build customer_ltv?" --top-k 8
```

AWS KB:

```bash
uv run python scripts/search.py --config kb_config.yaml --query "which tables feed gross margin metric?" --top-k 8
```

## Planner hooks (Claude/Strands)

Planner decides embedding/chunking strategy before ingestion:

```yaml
planner:
  mode: external_command
  external_command:
    command: uv run python scripts/agent_plan_stub.py
```

The command receives JSON on stdin and returns plan JSON on stdout.

## Bedrock bootstrap

One-time KB provisioning:

```bash
uv run python scripts/bootstrap_kb.py --config kb_config.yaml --wait
```

## Cron pattern

```cron
*/15 * * * * cd /tmp && rm -rf kb-run && git clone --depth=200 git@github.com:xdanny/KnowledgeBuilder.git kb-run && cd kb-run && uv sync --frozen && uv run python scripts/reindex.py --config kb_config.yaml --print-plan
```
