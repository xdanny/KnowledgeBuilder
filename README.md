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
cp kb_config.example.yaml kb_config.yaml
uv sync
```

Optional extras:

```bash
# Needed for local FAISS index + BM25 retrieval
uv sync --extra faiss

# Optional better local embeddings for FAISS
uv sync --extra local-embeddings
```

## Quickstart A: Local Mode (FAISS)

This is the fastest way to start.

### 1) Minimal `kb_config.yaml` for local mode

Use this as a starting point:

```yaml
aws:
  region: us-east-1
  source_bucket:
  source_prefix: kb-docs
  knowledge_base_id:
  data_source_id:
  glue_databases: []

repositories:
  - name: spark-jobs
    path: /absolute/path/to/your/spark-or-flink-repo
    git_ref: HEAD

state:
  backend: local
  local_path: .kb_state/state.json
  s3_bucket:
  s3_prefix: kb-state

backend:
  type: faiss
  aws_kb:
    start_ingestion_job: false
    wait_for_ingestion_job: false
  faiss:
    index_path: .kb_local/index.faiss
    metadata_path: .kb_local/metadata.json
    embedding_provider: hash
    embedding_model: sentence-transformers/all-MiniLM-L6-v2
    embedding_dimension: 768
    normalize_embeddings: true
    bm25_enabled: true
    rrf_k: 60
    rerank_enabled: true
    initial_retrieval_k: 150

planner:
  mode: heuristic
  external_command:
    command:

indexing:
  include_extensions: [".py", ".scala", ".sql", ".yaml", ".yml", ".json"]
  exclude_dirs: [".git", ".venv", "venv", "__pycache__", "target", "build"]
  max_chars_per_doc: 45000
  contextual_retrieval_enabled: true
  contextual_chunk_size_chars: 1800
  contextual_chunk_overlap_chars: 220
  contextualizer_mode: heuristic
  contextualizer_command:
  contextualizer_max_context_chars: 320
  graph_db_path: .kb_state/graph.db
  impact_reindex_enabled: true

bootstrap:
  knowledge_base_name:
  data_source_name:
  bedrock_role_arn:
  embedding_model_arn:
  vector_bucket_name:
  vector_index_name:
  vector_dimension: 1024
  vector_distance_metric: cosine
  chunk_lambda_arn:
  intermediate_s3_uri:
  context_enrichment_model_arn:
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

### 1) Set backend + state

In `kb_config.yaml`:

1. Set `backend.type: aws_kb`.
2. Set `state.backend: s3`.
3. Set `state.s3_bucket` (or reuse `aws.source_bucket`).

### 2) Fill bootstrap values

Set:

1. `bootstrap.knowledge_base_name`
2. `bootstrap.data_source_name`
3. `bootstrap.bedrock_role_arn`
4. `bootstrap.embedding_model_arn`
5. `bootstrap.vector_bucket_name`
6. `bootstrap.vector_index_name`
7. `aws.source_bucket`

### 3) Bootstrap once

```bash
uv run python scripts/bootstrap_kb.py --config kb_config.yaml --wait
```

Copy output IDs into config:

1. `aws.knowledge_base_id`
2. `aws.data_source_id`

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

Modes:

1. `heuristic`: zero-model-cost context generation.
2. `external_command`: your command receives JSON on stdin and returns `{"context":"..."}`.

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

1. `repositories[].path` exists and is a git repo.
2. `indexing.include_extensions` includes your file types.
3. `indexing.exclude_dirs` is not excluding too much.

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
5. Search: `scripts/search.py`
