# KnowledgeBuilder

KnowledgeBuilder builds a data-engineering knowledge base from code + Glue metadata with:

1. Incremental indexing by git commit and file hash.
2. Graph-based code understanding (symbols, calls, table refs, metric refs).
3. Structured markdown artifacts (`artifact`, `table`, `metric`, `architecture`).
4. Contextual Retrieval style chunking for better RAG quality.
5. Two backends:
- `faiss` (local development / local search)
- `aws_kb` (Bedrock Knowledge Base + S3 source docs)

## Architecture Summary

- Parser layer (`kb_indexer/extractor.py`):
  - Python AST symbols and call extraction.
  - SQL/Scala structural extraction and table/metric heuristics.
- Graph store (`kb_indexer/graph_store.py`):
  - SQLite graph tables for files, refs, symbols, call edges.
- Reindex orchestrator (`scripts/reindex.py`):
  - Reads git diff, applies hash-skip, updates graph, regenerates impacted docs.
  - Supports repo `path` or `git_url` (clone/pull before indexing).
- Retrieval:
  - Local hybrid retrieval in `faiss` mode (`scripts/search.py`).
  - Bedrock retrieval in `aws_kb` mode (`scripts/search.py`).

## Prerequisites

1. Python 3.11+.
2. `uv` installed.
3. Git available.
4. For AWS mode:
- AWS credentials configured.
- Access to S3, Glue, Bedrock Agent APIs, and optionally S3 Vectors bootstrap APIs.

## Install

```bash
cd /Users/dan/Documents/New\ project
cp kb_config.local.example.yaml kb_config.yaml
uv sync
```

Optional extras:

```bash
# Needed for local FAISS index + BM25 retrieval
uv sync --extra faiss

# Optional better local embeddings for FAISS
uv sync --extra local-embeddings

# Optional LiteLLM contextualizer mode
uv sync --extra llm-context
```

## Simplest Setup (Recommended)

Use one of the mode-specific templates instead of the full config:

1. Local mode:

```bash
cp kb_config.local.example.yaml kb_config.yaml
```

2. AWS mode:

```bash
cp kb_config.aws.example.yaml kb_config.yaml
```

Then edit only the required fields.

`mode` presets:

1. `mode: local` defaults to local state + FAISS backend.
2. `mode: aws` defaults to S3 state + AWS KB backend.

Single-repo shorthand:

1. `repo_path` and `repo_name` can replace the full `repositories:` list.
2. Or use `repo_git_url` (+ optional `repo_git_branch`, `repo_checkout_path`) instead of `repo_path`.
3. Use `repositories:` only when indexing multiple repos.

## Quickstart A: Local Mode (FAISS)

This is the fastest way to start.

### 1) Minimal `kb_config.yaml` for local mode

Use this as a starting point (or copy `kb_config.local.example.yaml`):

```yaml
mode: local
repo_path: /absolute/path/to/your/spark-or-flink-repo
repo_name: spark-jobs
```

Git source alternative:

```yaml
mode: local
repo_git_url: git@github.com:your-org/spark-jobs.git
repo_git_branch: main
repo_checkout_path: .kb_repos/spark-jobs
repo_name: spark-jobs
```

### 2) Run first index

```bash
uv run python scripts/reindex.py --config kb_config.yaml --print-plan
```

### 3) Verify generated local state

After the first run, expect:

1. `.kb_state/graph.db`
2. `.kb_state/state.json`
3. `.kb_local/index.faiss`
4. `.kb_local/metadata.json`

### 4) Query locally

```bash
uv run python scripts/search.py --config kb_config.yaml --query "which jobs write prod.orders?" --top-k 8
```

## Quickstart B: AWS Bedrock KB Mode

Use this when you want managed KB ingestion/retrieval.

### 1) Start from AWS template

```bash
cp kb_config.aws.example.yaml kb_config.yaml
```

### 2) Set required fields

1. `repo_path`
2. `aws.source_bucket`
3. `aws.knowledge_base_id`
4. `aws.data_source_id`
5. `state.s3_bucket`

### 3) Optional bootstrap once (if KB is not created yet)

Fill `bootstrap.*` values and run:

```bash
uv run python scripts/bootstrap_kb.py --config kb_config.yaml --wait
```

### 4) Run indexing

```bash
uv run python scripts/reindex.py --config kb_config.yaml --print-plan
```

### 5) Query Bedrock KB

```bash
uv run python scripts/search.py --config kb_config.yaml --query "which tables feed gross_margin?" --top-k 8
```

## What `reindex.py` actually does

Per repository:

1. Reads previous commit SHA from state store (S3 or local file).
2. Computes git changes since last run.
3. Applies file extension/directory filters.
4. Uses file hash to skip unchanged file contents.
5. Parses changed files for:
- symbols
- call edges
- table reads/writes
- metric formulas
6. Upserts file facts into SQLite graph store.
7. Regenerates:
- changed artifact docs
- impacted table docs
- impacted metric docs
- architecture overview doc
8. Sends docs to backend:
- local FAISS index, or
- S3 source docs + optional Bedrock ingestion job.
9. Saves new commit SHA to state store.

## Contextual Retrieval Controls

Config keys:

1. `indexing.contextual_retrieval_enabled`
2. `indexing.contextual_chunk_size_chars`
3. `indexing.contextual_chunk_overlap_chars`
4. `indexing.contextualizer_mode`
5. `indexing.contextualizer_command`
6. `indexing.contextualizer_model`
7. `indexing.contextualizer_cache_path`
8. `indexing.contextualizer_max_tokens`
9. `indexing.contextualizer_temperature`

Modes:

1. `heuristic`: zero-model-cost context generation.
2. `external_command`: your command receives JSON on stdin and returns `{"context":"..."}`.
3. `bedrock`: built-in Bedrock runtime call using `contextualizer_model`.
4. `litellm`: built-in LiteLLM call using `contextualizer_model`.

Notes:

1. Planner and contextualizer are separate systems.
2. Planner decides ingest strategy once per run.
3. Contextualizer enriches every chunk and can use a cheaper model.
4. `contextualizer_cache_path` avoids paying again for unchanged chunks.

Stub script:

```bash
uv run python scripts/contextualize_chunk_stub.py
```

## Planner Integration

Planner runs once per indexing execution.

1. `planner.mode: heuristic` uses built-in rules.
2. `planner.mode: external_command` calls your agent command.

External planner contract:

1. stdin JSON payload with repo/backend/indexing context.
2. stdout JSON with:
- `backend`
- `embedding_provider`
- `embedding_model`
- `chunk_size_chars`
- `chunk_overlap_chars`
- `notes`

Stub script:

```bash
uv run python scripts/agent_plan_stub.py
```

## Planner vs Contextualizer (Important)

1. Planner (`planner.*`) is for orchestration decisions:
- backend selection
- embedding provider/model choice
- chunk size/overlap
2. Contextualizer (`indexing.contextualizer_*`) is for per-chunk context generation.
3. You can keep planner heuristic and still run LLM contextualization.
4. Recommended cost setup:
- planner: `heuristic`
- contextualizer mode: `bedrock` with a small model, or `litellm` with a low-cost model.

## Cron / Automation

For stateless periodic runs:

```cron
*/15 * * * * cd /tmp && rm -rf kb-run && git clone --depth=200 git@github.com:xdanny/KnowledgeBuilder.git kb-run && cd kb-run && uv sync --frozen && uv run python scripts/reindex.py --config kb_config.yaml --print-plan
```

Recommended with cron:

1. Use `state.backend: s3`.
2. Keep `impact_reindex_enabled: true`.
3. Use one config per environment (dev/stage/prod).

## Troubleshooting

### `ModuleNotFoundError` (yaml, faiss, rank_bm25)

Run:

```bash
uv sync
uv sync --extra faiss
```

### No files indexed

Check:

1. You configured either `repositories[].path` or `repositories[].git_url` (or shorthand `repo_path` / `repo_git_url`).
2. If using `git_url`, verify credentials and network access for clone/pull.
3. `indexing.include_extensions` includes your file types.
4. `indexing.exclude_dirs` is not excluding too much.

### AWS ingestion not starting

Check:

1. `backend.type` is `aws_kb`.
2. `aws.knowledge_base_id` and `aws.data_source_id` are set.
3. `backend.aws_kb.start_ingestion_job: true`.

### Glue context missing

Check:

1. `aws.glue_databases` is populated.
2. table names include database prefix and match Glue catalog naming.

## Key Files

1. Config schema: `kb_indexer/settings.py`
2. Reindex flow: `scripts/reindex.py`
3. Parser/extraction: `kb_indexer/extractor.py`
4. Graph store: `kb_indexer/graph_store.py`
5. Minimal local config: `kb_config.local.example.yaml`
6. Minimal AWS config: `kb_config.aws.example.yaml`
7. Full config template: `kb_config.example.yaml`
8. Search: `scripts/search.py`
