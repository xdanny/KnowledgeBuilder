# EMR/Flink Knowledge Base Indexer (AWS Bedrock + S3 Vectors)

Simple workflow:

1. One-time bootstrap creates:
   - S3 source bucket (documents)
   - S3 Vector bucket + index
   - Bedrock Knowledge Base (vector store = S3 Vectors)
   - S3 data source with optional post-chunking Lambda
2. Incremental runs:
   - read last indexed commit SHA from S3
   - `git diff` to find changed/deleted files
   - build enriched docs from Spark/Flink code + Glue Catalog context
   - upload only changed docs, delete removed docs
   - start Bedrock ingestion job
   - persist new SHA back to S3

## Prerequisites

- Python 3.11+
- `uv`
- AWS credentials with access to:
  - `bedrock-agent:*` (or narrowed create/list/get/start ingestion actions)
  - `s3vectors:*` (or narrowed bucket/index actions)
  - `s3:*` on your source bucket
  - `glue:GetTables` for configured Glue databases

## Setup

```bash
cp kb_config.example.yaml kb_config.yaml
uv sync
```

Edit `kb_config.yaml`:
- fill AWS identifiers/ARNs
- set `bootstrap.*` values for first-time provisioning
- set `aws.knowledge_base_id` and `aws.data_source_id` after bootstrap

## One-Time Bootstrap

```bash
uv run python scripts/bootstrap_kb.py --config kb_config.yaml --wait
```

Command output includes `knowledge_base_id` and `data_source_id`. Put them in `kb_config.yaml`.

## Incremental Reindex

```bash
uv run python scripts/reindex.py --config kb_config.yaml --repo .
```

Force a full rebuild:

```bash
uv run python scripts/reindex.py --config kb_config.yaml --repo . --full
```

## Local Cron Pattern

If cron clones a fresh repo each run, state still works because last indexed SHA is stored in S3:

```cron
*/15 * * * * cd /tmp && rm -rf de-kb-run && git clone --depth=200 git@github.com:ORG/REPO.git de-kb-run && cd de-kb-run && uv sync --frozen && uv run python scripts/reindex.py --config kb_config.yaml --repo .
```

## Lambda Chunker

File: `lambda/chunker/app.py`

- Implements Bedrock custom transformation for `POST_CHUNKING`
- Splits large content into semantic-ish chunks with overlap
- Adds metadata (`chunk_no`, `table_refs`) to each chunk

Package and deploy:

```bash
cd lambda/chunker
zip -r ../../chunker.zip app.py
```

Then create/update a Lambda function from `chunker.zip` and set:
- `bootstrap.chunk_lambda_arn`
- `bootstrap.intermediate_s3_uri`

## Notes

- The extractor uses heuristics for table reads/writes; tune regex in `kb_indexer/extractor.py`.
- Glue context enrichment only includes databases listed in `aws.glue_databases`.
- Bedrock ingestion remains incremental because only changed source objects are updated.
