#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import boto3

from kb_indexer.backends import build_backend
from kb_indexer.contextual_retrieval import contextualize_document
from kb_indexer.extractor import (
    StructuredFacts,
    analyze_artifact,
    build_architecture_document,
    build_artifact_document,
    build_metric_document,
    build_table_document,
    compute_file_hash,
    iter_files,
    load_glue_catalog,
)
from kb_indexer.git_utils import (
    commit_exists,
    ensure_repo_checkout,
    list_all_files,
    parse_git_changes,
    resolve_sha,
)
from kb_indexer.graph_store import GraphStore
from kb_indexer.planner import build_plan
from kb_indexer.settings import AppSettings, RepoSettings, load_settings
from kb_indexer.state_store import LocalFileStateStore, S3StateStore, StateStore


@dataclass
class RepoRunResult:
    repo_name: str
    repo_path: str
    saved_sha: str | None
    head_sha: str
    changed_files: int
    deleted_files: int
    indexed_docs: int
    parsed_files: int
    hash_skipped_files: int
    impacted_tables: int
    impacted_metrics: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Incremental multi-repo indexing with graph-based impact reindexing "
            "for AWS Bedrock KB or local FAISS backend."
        )
    )
    parser.add_argument("--config", default="kb_config.yaml", help="Path to config YAML.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Ignore saved state and rebuild graph/docs for all files in each repository.",
    )
    parser.add_argument(
        "--repos",
        nargs="*",
        default=None,
        help="Optional repository names from config to run. Defaults to all configured repos.",
    )
    parser.add_argument(
        "--print-plan",
        action="store_true",
        help="Print planner output before running ingestion.",
    )
    return parser.parse_args()


def build_state_store(settings: AppSettings, s3_client: Any) -> StateStore:
    if settings.state.backend == "local":
        return LocalFileStateStore(settings.state.local_path)
    if settings.state.backend == "s3":
        if not settings.state.s3_bucket:
            raise ValueError("state.s3_bucket is required when state.backend is s3.")
        return S3StateStore(s3_client=s3_client, bucket=settings.state.s3_bucket)
    raise ValueError(f"Unsupported state backend: {settings.state.backend}")


def _compute_changes(
    repo_root: Path,
    saved_sha: str | None,
    head_sha: str,
    full: bool,
) -> tuple[list[str], list[str]]:
    changed_paths: list[str] = []
    deleted_paths: list[str] = []

    if not full and saved_sha and commit_exists(repo_root, saved_sha):
        changes = parse_git_changes(repo_root, saved_sha, head_sha)
        for change in changes:
            if change.status == "D" and change.old_path:
                deleted_paths.append(change.old_path)
            elif change.status == "R":
                if change.old_path:
                    deleted_paths.append(change.old_path)
                if change.new_path:
                    changed_paths.append(change.new_path)
            elif change.new_path:
                changed_paths.append(change.new_path)
        return changed_paths, deleted_paths

    return list_all_files(repo_root), []


def _filter_allowed(
    repo_root: Path,
    include_ext: list[str],
    exclude_dirs: list[str],
    paths: list[str],
) -> list[str]:
    allowed_paths = {
        path.relative_to(repo_root).as_posix()
        for path in iter_files(repo_root, include_ext, exclude_dirs)
    }
    return sorted({path for path in paths if path in allowed_paths})


def _doc_source_path(repo_name: str, repo_relative_path: str) -> str:
    return f"{repo_name}/{repo_relative_path}"


def _namespace_doc_path(repo_name: str, source_path: str) -> str:
    if source_path.startswith(f"{repo_name}/"):
        return source_path
    return _doc_source_path(repo_name, source_path)


def _with_contextual_sections(source_path: str, content: str, settings: AppSettings) -> str:
    if not settings.indexing.contextual_retrieval_enabled:
        return content
    chunks = contextualize_document(source_path=source_path, document_text=content, settings=settings)
    lines = [content, "", "## Contextual Retrieval Chunks"]
    for chunk in chunks:
        lines.append(f"### Chunk {chunk.chunk_index}")
        lines.append("```text")
        lines.append(chunk.contextual_text)
        lines.append("```")
    return "\n".join(lines)


def _collect_impacted_from_facts(
    facts: dict[str, Any] | None,
    impacted_tables: set[str],
    impacted_metrics: set[str],
) -> None:
    if not facts:
        return
    for table_name in facts.get("reads", []):
        impacted_tables.add(str(table_name).lower())
    for table_name in facts.get("writes", []):
        impacted_tables.add(str(table_name).lower())
    for metric in facts.get("metrics", []):
        metric_name = str(metric.get("name", "")).lower().strip()
        if metric_name:
            impacted_metrics.add(metric_name)


def _build_structured_docs_from_graph(
    *,
    repo_name: str,
    commit_sha: str,
    graph_store: GraphStore,
    glue_catalog: dict[str, dict[str, Any]],
    impacted_tables: set[str],
    impacted_metrics: set[str],
    force_all: bool,
) -> tuple[list[Any], list[str]]:
    docs = []
    stale_paths: list[str] = []

    table_candidates = set(graph_store.list_tables(repo_name)) if force_all else set(impacted_tables)
    metric_candidates = set(graph_store.list_metrics(repo_name)) if force_all else set(impacted_metrics)

    for table_name in sorted(table_candidates):
        usage = graph_store.get_table_usage(repo_name, table_name)
        table_doc_path = f"{repo_name}/entities/tables/{table_name}.md"
        if not usage.get("reads") and not usage.get("writes"):
            stale_paths.append(table_doc_path)
            continue
        docs.append(
            build_table_document(
                repo_name=repo_name,
                table_name=table_name,
                usage=usage,
                glue_catalog=glue_catalog,
                commit_sha=commit_sha,
            )
        )

    for metric_name in sorted(metric_candidates):
        definitions = graph_store.get_metric_definitions(repo_name, metric_name)
        metric_doc_path = f"{repo_name}/entities/metrics/{metric_name}.md"
        if not definitions:
            stale_paths.append(metric_doc_path)
            continue
        docs.append(
            build_metric_document(
                repo_name=repo_name,
                metric_name=metric_name,
                definitions=definitions,
                commit_sha=commit_sha,
            )
        )

    all_tables = {
        table_name: graph_store.get_table_usage(repo_name, table_name)
        for table_name in graph_store.list_tables(repo_name)
    }
    all_metrics = {
        metric_name: graph_store.get_metric_definitions(repo_name, metric_name)
        for metric_name in graph_store.list_metrics(repo_name)
    }
    architecture_doc = build_architecture_document(
        repo_name=repo_name,
        commit_sha=commit_sha,
        facts=StructuredFacts(
            tables=all_tables,
            metrics=all_metrics,
            top_files=graph_store.get_top_files(repo_name, limit=20),
            call_edges=graph_store.get_call_edges(repo_name, limit=200),
            file_count=graph_store.get_repo_file_count(repo_name),
        ),
    )
    docs.append(architecture_doc)
    return docs, stale_paths


def _run_repo(
    *,
    repo: RepoSettings,
    settings: AppSettings,
    state_store: StateStore,
    glue_catalog: dict[str, dict[str, Any]],
    backend: Any,
    graph_store: GraphStore,
    full: bool,
) -> RepoRunResult:
    repo_root = Path(repo.path).resolve()
    head_sha = resolve_sha(repo_root, repo.git_ref)
    saved_sha = None if full else state_store.read(repo.state_key or "")

    changed_paths, deleted_paths = _compute_changes(repo_root, saved_sha, head_sha, full)
    all_allowed_paths = sorted(
        path.relative_to(repo_root).as_posix()
        for path in iter_files(
            repo_root,
            settings.indexing.include_extensions,
            settings.indexing.exclude_dirs,
        )
    )
    changed_filtered = _filter_allowed(
        repo_root=repo_root,
        include_ext=settings.indexing.include_extensions,
        exclude_dirs=settings.indexing.exclude_dirs,
        paths=changed_paths,
    )

    if full:
        backend.delete_by_prefix([f"{repo.name}/"])
        graph_store.clear_repo(repo.name)
        changed_filtered = list(all_allowed_paths)
        deleted_paths = []

    impacted_tables: set[str] = set()
    impacted_metrics: set[str] = set()
    docs = []
    parsed_files = 0
    hash_skipped_files = 0

    deleted_filtered = sorted(set(deleted_paths))
    if deleted_filtered:
        deleted_namespaced = [_doc_source_path(repo.name, path) for path in deleted_filtered]
        backend.delete_documents(deleted_namespaced)
        for path in deleted_filtered:
            facts = graph_store.delete_file(repo.name, path)
            _collect_impacted_from_facts(facts, impacted_tables, impacted_metrics)

    for relative_path in changed_filtered:
        file_path = repo_root / relative_path
        if not file_path.exists():
            continue
        content_hash = compute_file_hash(file_path)
        previous_hash = graph_store.get_file_hash(repo.name, relative_path)
        if (
            settings.indexing.impact_reindex_enabled
            and not full
            and previous_hash == content_hash
        ):
            hash_skipped_files += 1
            continue

        analysis = analyze_artifact(
            repo_root=repo_root,
            file_path=file_path,
            max_chars_per_doc=settings.indexing.max_chars_per_doc,
        )
        graph_store.upsert_file_analysis(
            repo=repo.name,
            source_path=relative_path,
            content_hash=content_hash,
            commit_sha=head_sha,
            analysis=analysis,
        )
        parsed_files += 1
        for table_name in analysis.reads + analysis.writes:
            impacted_tables.add(table_name.lower())
        for metric in analysis.metrics:
            impacted_metrics.add(metric.name.lower())

        artifact_doc = build_artifact_document(analysis=analysis, glue_catalog=glue_catalog)
        artifact_doc.source_path = _namespace_doc_path(repo.name, artifact_doc.source_path)
        if settings.backend.type == "aws_kb":
            artifact_doc.content = _with_contextual_sections(
                source_path=artifact_doc.source_path,
                content=artifact_doc.content,
                settings=settings,
            )
        docs.append(artifact_doc)

    needs_structured_refresh = full or bool(changed_filtered) or bool(deleted_filtered)
    if needs_structured_refresh:
        force_all_structured = full or (not settings.indexing.impact_reindex_enabled)
        structured_docs, stale_paths = _build_structured_docs_from_graph(
            repo_name=repo.name,
            commit_sha=head_sha,
            graph_store=graph_store,
            glue_catalog=glue_catalog,
            impacted_tables=impacted_tables,
            impacted_metrics=impacted_metrics,
            force_all=force_all_structured,
        )
        if stale_paths:
            backend.delete_documents(stale_paths)
        if settings.backend.type == "aws_kb" and settings.indexing.contextual_retrieval_enabled:
            for doc in structured_docs:
                doc.content = _with_contextual_sections(
                    source_path=doc.source_path,
                    content=doc.content,
                    settings=settings,
                )
        docs.extend(structured_docs)

    if docs:
        backend.upsert_documents(docs)
    if repo.state_key:
        state_store.write(repo.state_key, head_sha)

    return RepoRunResult(
        repo_name=repo.name,
        repo_path=str(repo_root),
        saved_sha=saved_sha,
        head_sha=head_sha,
        changed_files=len(changed_filtered),
        deleted_files=len(deleted_filtered),
        indexed_docs=len(docs),
        parsed_files=parsed_files,
        hash_skipped_files=hash_skipped_files,
        impacted_tables=len(impacted_tables),
        impacted_metrics=len(impacted_metrics),
    )


def _prepare_repo(repo: RepoSettings) -> RepoSettings:
    if not repo.git_url:
        return repo
    checkout = ensure_repo_checkout(
        git_url=repo.git_url,
        checkout_path=Path(repo.path),
        git_branch=repo.git_branch,
        git_ref=repo.git_ref,
    )
    return replace(repo, path=str(checkout))


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    session = boto3.Session(region_name=settings.aws.region)
    s3_client = session.client("s3")
    glue_client = session.client("glue")
    bedrock_agent = session.client("bedrock-agent")

    selected_repos = settings.repositories
    if args.repos:
        selected = set(args.repos)
        selected_repos = [repo for repo in settings.repositories if repo.name in selected]
        if not selected_repos:
            raise ValueError(f"No repositories matched --repos {args.repos}")
    selected_repos = [_prepare_repo(repo) for repo in selected_repos]

    context = {
        "repo_count": len(selected_repos),
        "backend": settings.backend.type,
        "repos": [
            {"name": repo.name, "path": repo.path, "git_url": repo.git_url, "git_branch": repo.git_branch}
            for repo in selected_repos
        ],
    }
    plan = build_plan(settings, context)
    if args.print_plan:
        print(
            json.dumps(
                {
                    "backend": plan.backend,
                    "embedding_provider": plan.embedding_provider,
                    "embedding_model": plan.embedding_model,
                    "chunk_size_chars": plan.chunk_size_chars,
                    "chunk_overlap_chars": plan.chunk_overlap_chars,
                    "notes": plan.notes,
                },
                indent=2,
            )
        )

    # The planner can override runtime embedding/chunking parameters.
    settings.indexing.contextual_chunk_size_chars = plan.chunk_size_chars
    settings.indexing.contextual_chunk_overlap_chars = plan.chunk_overlap_chars
    if settings.backend.type == "faiss":
        settings.backend.faiss.embedding_provider = plan.embedding_provider
        settings.backend.faiss.embedding_model = plan.embedding_model

    state_store = build_state_store(settings, s3_client=s3_client)
    backend = build_backend(
        settings=settings,
        s3_client=s3_client,
        bedrock_agent_client=bedrock_agent,
    )
    glue_catalog = load_glue_catalog(glue_client, settings.aws.glue_databases)
    graph_store = GraphStore(settings.indexing.graph_db_path)

    repo_results: list[RepoRunResult] = []
    try:
        for repo in selected_repos:
            repo_results.append(
                _run_repo(
                    repo=repo,
                    settings=settings,
                    state_store=state_store,
                    glue_catalog=glue_catalog,
                    backend=backend,
                    graph_store=graph_store,
                    full=args.full,
                )
            )
    finally:
        graph_store.close()

    backend_result = backend.finalize()
    print(
        json.dumps(
            {
                "planner": {
                    "backend": plan.backend,
                    "embedding_provider": plan.embedding_provider,
                    "embedding_model": plan.embedding_model,
                    "chunk_size_chars": plan.chunk_size_chars,
                    "chunk_overlap_chars": plan.chunk_overlap_chars,
                    "notes": plan.notes,
                },
                "repositories": [result.__dict__ for result in repo_results],
                "backend_result": backend_result,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
