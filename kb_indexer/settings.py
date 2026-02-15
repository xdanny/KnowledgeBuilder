from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AwsSettings:
    region: str
    source_bucket: str
    source_prefix: str = "kb-docs"
    state_key: str = "kb-state/last_indexed_sha.txt"
    knowledge_base_id: str | None = None
    data_source_id: str | None = None
    glue_databases: list[str] = field(default_factory=list)


@dataclass
class BootstrapSettings:
    knowledge_base_name: str | None = None
    data_source_name: str | None = None
    bedrock_role_arn: str | None = None
    embedding_model_arn: str | None = None
    vector_bucket_name: str | None = None
    vector_index_name: str | None = None
    vector_dimension: int = 1024
    vector_distance_metric: str = "cosine"
    chunk_lambda_arn: str | None = None
    intermediate_s3_uri: str | None = None
    context_enrichment_model_arn: str | None = None


@dataclass
class IndexingSettings:
    include_extensions: list[str] = field(
        default_factory=lambda: [".py", ".scala", ".sql", ".yaml", ".yml", ".json"]
    )
    exclude_dirs: list[str] = field(
        default_factory=lambda: [".git", ".venv", "venv", "__pycache__", "target", "build"]
    )
    max_chars_per_doc: int = 45000
    start_ingestion_job: bool = True
    wait_for_ingestion_job: bool = False


@dataclass
class AppSettings:
    aws: AwsSettings
    bootstrap: BootstrapSettings = field(default_factory=BootstrapSettings)
    indexing: IndexingSettings = field(default_factory=IndexingSettings)


def _get(section: dict[str, Any], key: str, default: Any) -> Any:
    value = section.get(key, default)
    return default if value is None else value


def load_settings(config_path: str | Path) -> AppSettings:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    aws_raw = raw.get("aws", {})
    bootstrap_raw = raw.get("bootstrap", {})
    indexing_raw = raw.get("indexing", {})

    aws = AwsSettings(
        region=aws_raw["region"],
        source_bucket=aws_raw["source_bucket"],
        source_prefix=_get(aws_raw, "source_prefix", "kb-docs"),
        state_key=_get(aws_raw, "state_key", "kb-state/last_indexed_sha.txt"),
        knowledge_base_id=aws_raw.get("knowledge_base_id"),
        data_source_id=aws_raw.get("data_source_id"),
        glue_databases=_get(aws_raw, "glue_databases", []),
    )
    bootstrap = BootstrapSettings(
        knowledge_base_name=bootstrap_raw.get("knowledge_base_name"),
        data_source_name=bootstrap_raw.get("data_source_name"),
        bedrock_role_arn=bootstrap_raw.get("bedrock_role_arn"),
        embedding_model_arn=bootstrap_raw.get("embedding_model_arn"),
        vector_bucket_name=bootstrap_raw.get("vector_bucket_name"),
        vector_index_name=bootstrap_raw.get("vector_index_name"),
        vector_dimension=int(_get(bootstrap_raw, "vector_dimension", 1024)),
        vector_distance_metric=str(_get(bootstrap_raw, "vector_distance_metric", "cosine")),
        chunk_lambda_arn=bootstrap_raw.get("chunk_lambda_arn"),
        intermediate_s3_uri=bootstrap_raw.get("intermediate_s3_uri"),
        context_enrichment_model_arn=bootstrap_raw.get("context_enrichment_model_arn"),
    )
    indexing = IndexingSettings(
        include_extensions=[str(x) for x in _get(indexing_raw, "include_extensions", [])]
        or [".py", ".scala", ".sql", ".yaml", ".yml", ".json"],
        exclude_dirs=[str(x) for x in _get(indexing_raw, "exclude_dirs", [])]
        or [".git", ".venv", "venv", "__pycache__", "target", "build"],
        max_chars_per_doc=int(_get(indexing_raw, "max_chars_per_doc", 45000)),
        start_ingestion_job=bool(_get(indexing_raw, "start_ingestion_job", True)),
        wait_for_ingestion_job=bool(_get(indexing_raw, "wait_for_ingestion_job", False)),
    )
    return AppSettings(aws=aws, bootstrap=bootstrap, indexing=indexing)
