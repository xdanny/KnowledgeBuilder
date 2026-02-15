from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AwsSettings:
    region: str = "us-east-1"
    source_bucket: str | None = None
    source_prefix: str = "kb-docs"
    knowledge_base_id: str | None = None
    data_source_id: str | None = None
    glue_databases: list[str] = field(default_factory=list)


@dataclass
class RepoSettings:
    name: str
    path: str
    git_ref: str = "HEAD"
    state_key: str | None = None


@dataclass
class IndexingSettings:
    include_extensions: list[str] = field(
        default_factory=lambda: [".py", ".scala", ".sql", ".yaml", ".yml", ".json"]
    )
    exclude_dirs: list[str] = field(
        default_factory=lambda: [".git", ".venv", "venv", "__pycache__", "target", "build"]
    )
    max_chars_per_doc: int = 45000
    contextual_retrieval_enabled: bool = True
    contextual_chunk_size_chars: int = 1800
    contextual_chunk_overlap_chars: int = 220
    contextualizer_mode: str = "heuristic"
    contextualizer_command: str | None = None
    contextualizer_max_context_chars: int = 320


@dataclass
class StateSettings:
    backend: str = "s3"
    local_path: str = ".kb_state/state.json"
    s3_bucket: str | None = None
    s3_prefix: str = "kb-state"


@dataclass
class AwsKbBackendSettings:
    start_ingestion_job: bool = True
    wait_for_ingestion_job: bool = False


@dataclass
class FaissBackendSettings:
    index_path: str = ".kb_local/index.faiss"
    metadata_path: str = ".kb_local/metadata.json"
    embedding_provider: str = "hash"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 768
    normalize_embeddings: bool = True
    bm25_enabled: bool = True
    rrf_k: int = 60
    rerank_enabled: bool = True
    initial_retrieval_k: int = 150


@dataclass
class BackendSettings:
    type: str = "aws_kb"
    aws_kb: AwsKbBackendSettings = field(default_factory=AwsKbBackendSettings)
    faiss: FaissBackendSettings = field(default_factory=FaissBackendSettings)


@dataclass
class PlannerExternalCommandSettings:
    command: str | None = None


@dataclass
class PlannerSettings:
    mode: str = "heuristic"
    external_command: PlannerExternalCommandSettings = field(
        default_factory=PlannerExternalCommandSettings
    )


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
class AppSettings:
    aws: AwsSettings = field(default_factory=AwsSettings)
    bootstrap: BootstrapSettings = field(default_factory=BootstrapSettings)
    repositories: list[RepoSettings] = field(default_factory=list)
    indexing: IndexingSettings = field(default_factory=IndexingSettings)
    state: StateSettings = field(default_factory=StateSettings)
    backend: BackendSettings = field(default_factory=BackendSettings)
    planner: PlannerSettings = field(default_factory=PlannerSettings)


def _get(section: dict[str, Any], key: str, default: Any) -> Any:
    value = section.get(key, default)
    return default if value is None else value


def _normalize_repositories(raw: dict[str, Any]) -> list[RepoSettings]:
    repos_raw = raw.get("repositories")
    if not repos_raw:
        return [RepoSettings(name="default", path=".", git_ref="HEAD")]

    repos: list[RepoSettings] = []
    for idx, item in enumerate(repos_raw, start=1):
        name = str(item.get("name") or f"repo-{idx}")
        path = str(item.get("path") or ".")
        git_ref = str(item.get("git_ref") or "HEAD")
        state_key = item.get("state_key")
        repos.append(RepoSettings(name=name, path=path, git_ref=git_ref, state_key=state_key))
    return repos


def _state_key_for_repo(settings: AppSettings, repo: RepoSettings) -> str:
    if repo.state_key:
        return repo.state_key
    prefix = settings.state.s3_prefix.rstrip("/")
    return f"{prefix}/{repo.name}/last_indexed_sha.txt"


def load_settings(config_path: str | Path) -> AppSettings:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    aws_raw = raw.get("aws", {})
    indexing_raw = raw.get("indexing", {})
    state_raw = raw.get("state", {})
    backend_raw = raw.get("backend", {})
    planner_raw = raw.get("planner", {})
    bootstrap_raw = raw.get("bootstrap", {})

    aws = AwsSettings(
        region=str(_get(aws_raw, "region", "us-east-1")),
        source_bucket=aws_raw.get("source_bucket"),
        source_prefix=str(_get(aws_raw, "source_prefix", "kb-docs")),
        knowledge_base_id=aws_raw.get("knowledge_base_id"),
        data_source_id=aws_raw.get("data_source_id"),
        glue_databases=[str(x) for x in _get(aws_raw, "glue_databases", [])],
    )

    indexing = IndexingSettings(
        include_extensions=[str(x) for x in _get(indexing_raw, "include_extensions", [])]
        or [".py", ".scala", ".sql", ".yaml", ".yml", ".json"],
        exclude_dirs=[str(x) for x in _get(indexing_raw, "exclude_dirs", [])]
        or [".git", ".venv", "venv", "__pycache__", "target", "build"],
        max_chars_per_doc=int(_get(indexing_raw, "max_chars_per_doc", 45000)),
        contextual_retrieval_enabled=bool(
            _get(indexing_raw, "contextual_retrieval_enabled", True)
        ),
        contextual_chunk_size_chars=int(
            _get(indexing_raw, "contextual_chunk_size_chars", 1800)
        ),
        contextual_chunk_overlap_chars=int(
            _get(indexing_raw, "contextual_chunk_overlap_chars", 220)
        ),
        contextualizer_mode=str(_get(indexing_raw, "contextualizer_mode", "heuristic")).lower(),
        contextualizer_command=_get(indexing_raw, "contextualizer_command", None),
        contextualizer_max_context_chars=int(
            _get(indexing_raw, "contextualizer_max_context_chars", 320)
        ),
    )

    state = StateSettings(
        backend=str(_get(state_raw, "backend", "s3")).lower(),
        local_path=str(_get(state_raw, "local_path", ".kb_state/state.json")),
        s3_bucket=state_raw.get("s3_bucket"),
        s3_prefix=str(_get(state_raw, "s3_prefix", "kb-state")),
    )

    backend = BackendSettings(
        type=str(_get(backend_raw, "type", "aws_kb")).lower(),
        aws_kb=AwsKbBackendSettings(
            start_ingestion_job=bool(
                _get(
                    _get(backend_raw, "aws_kb", {}),
                    "start_ingestion_job",
                    True,
                )
            ),
            wait_for_ingestion_job=bool(
                _get(
                    _get(backend_raw, "aws_kb", {}),
                    "wait_for_ingestion_job",
                    False,
                )
            ),
        ),
        faiss=FaissBackendSettings(
            index_path=str(_get(_get(backend_raw, "faiss", {}), "index_path", ".kb_local/index.faiss")),
            metadata_path=str(
                _get(_get(backend_raw, "faiss", {}), "metadata_path", ".kb_local/metadata.json")
            ),
            embedding_provider=str(
                _get(_get(backend_raw, "faiss", {}), "embedding_provider", "hash")
            ).lower(),
            embedding_model=str(
                _get(
                    _get(backend_raw, "faiss", {}),
                    "embedding_model",
                    "sentence-transformers/all-MiniLM-L6-v2",
                )
            ),
            embedding_dimension=int(
                _get(_get(backend_raw, "faiss", {}), "embedding_dimension", 768)
            ),
            normalize_embeddings=bool(
                _get(_get(backend_raw, "faiss", {}), "normalize_embeddings", True)
            ),
            bm25_enabled=bool(_get(_get(backend_raw, "faiss", {}), "bm25_enabled", True)),
            rrf_k=int(_get(_get(backend_raw, "faiss", {}), "rrf_k", 60)),
            rerank_enabled=bool(_get(_get(backend_raw, "faiss", {}), "rerank_enabled", True)),
            initial_retrieval_k=int(
                _get(_get(backend_raw, "faiss", {}), "initial_retrieval_k", 150)
            ),
        ),
    )

    planner = PlannerSettings(
        mode=str(_get(planner_raw, "mode", "heuristic")).lower(),
        external_command=PlannerExternalCommandSettings(
            command=_get(_get(planner_raw, "external_command", {}), "command", None)
        ),
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

    settings = AppSettings(
        aws=aws,
        bootstrap=bootstrap,
        repositories=_normalize_repositories(raw),
        indexing=indexing,
        state=state,
        backend=backend,
        planner=planner,
    )

    # Backward compatibility with older single-key state config.
    legacy_state_key = aws_raw.get("state_key")
    if legacy_state_key and settings.repositories:
        first_repo = settings.repositories[0]
        if first_repo.state_key is None:
            first_repo.state_key = str(legacy_state_key)

    if settings.state.s3_bucket is None:
        settings.state.s3_bucket = settings.aws.source_bucket

    for repo in settings.repositories:
        if repo.state_key is None:
            repo.state_key = _state_key_for_repo(settings, repo)

    return settings
