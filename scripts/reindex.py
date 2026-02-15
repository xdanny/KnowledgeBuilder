#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3

from kb_indexer.backends import build_backend
from kb_indexer.contextual_retrieval import contextualize_document
from kb_indexer.extractor import (
    analyze_artifact,
    build_artifact_document,
    build_structured_documents,
    iter_files,
    load_glue_catalog,
)
from kb_indexer.git_utils import commit_exists, list_all_files, parse_git_changes, resolve_sha
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Incremental multi-repo indexing for AWS Bedrock KB or local FAISS backend."
        )
    )
    parser.add_argument("--config", default="kb_config.yaml", help="Path to config YAML.")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Ignore saved state and index all tracked files for all repositories.",
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
    changed_paths: list[str],
) -> list[str]:
    allowed_paths = {
        path.relative_to(repo_root).as_posix()
        for path in iter_files(repo_root, include_ext, exclude_dirs)
    }
    return sorted({path for path in changed_paths if path in allowed_paths})


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


def _run_repo(
    *,
    repo: RepoSettings,
    settings: AppSettings,
    state_store: StateStore,
    glue_catalog: dict[str, dict[str, Any]],
    backend: Any,
    full: bool,
) -> RepoRunResult:
    repo_root = Path(repo.path).resolve()
    head_sha = resolve_sha(repo_root, repo.git_ref)
    saved_sha = None if full else state_store.read(repo.state_key or "")

    changed_paths, deleted_paths = _compute_changes(repo_root, saved_sha, head_sha, full)
    changed_filtered = _filter_allowed(
        repo_root=repo_root,
        include_ext=settings.indexing.include_extensions,
        exclude_dirs=settings.indexing.exclude_dirs,
        changed_paths=changed_paths,
    )
    all_allowed_paths = sorted(
        path.relative_to(repo_root).as_posix()
        for path in iter_files(
            repo_root,
            settings.indexing.include_extensions,
            settings.indexing.exclude_dirs,
        )
    )

    artifact_docs = []
    for relative_path in changed_filtered:
        file_path = repo_root / relative_path
        if not file_path.exists():
            continue
        analysis = analyze_artifact(
            repo_root=repo_root,
            file_path=file_path,
            max_chars_per_doc=settings.indexing.max_chars_per_doc,
        )
        artifact_doc = build_artifact_document(analysis=analysis, glue_catalog=glue_catalog)
        artifact_doc.source_path = _namespace_doc_path(repo.name, artifact_doc.source_path)
        if settings.backend.type == "aws_kb":
            artifact_doc.content = _with_contextual_sections(
                source_path=artifact_doc.source_path,
                content=artifact_doc.content,
                settings=settings,
            )
        artifact_docs.append(artifact_doc)

    deleted_namespaced = [_doc_source_path(repo.name, path) for path in sorted(set(deleted_paths))]
    if deleted_namespaced:
        backend.delete_documents(deleted_namespaced)

    # Structured docs are rebuilt from current repo state to keep table/metric/architecture docs authoritative.
    structured_docs = []
    if changed_filtered or deleted_namespaced or full:
        full_analyses = []
        for relative_path in all_allowed_paths:
            file_path = repo_root / relative_path
            if not file_path.exists():
                continue
            full_analyses.append(
                analyze_artifact(
                    repo_root=repo_root,
                    file_path=file_path,
                    max_chars_per_doc=settings.indexing.max_chars_per_doc,
                )
            )
        structured_docs = build_structured_documents(
            repo_name=repo.name,
            analyses=full_analyses,
            glue_catalog=glue_catalog,
            commit_sha=head_sha,
        )
        if settings.backend.type == "aws_kb" and settings.indexing.contextual_retrieval_enabled:
            for doc in structured_docs:
                doc.content = _with_contextual_sections(
                    source_path=doc.source_path,
                    content=doc.content,
                    settings=settings,
                )
        backend.delete_by_prefix([f"{repo.name}/entities/"])

    docs = artifact_docs + structured_docs
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
        deleted_files=len(deleted_namespaced),
        indexed_docs=len(docs),
    )


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

    context = {
        "repo_count": len(selected_repos),
        "backend": settings.backend.type,
        "repos": [{"name": repo.name, "path": repo.path} for repo in selected_repos],
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

    repo_results: list[RepoRunResult] = []
    for repo in selected_repos:
        repo_results.append(
            _run_repo(
                repo=repo,
                settings=settings,
                state_store=state_store,
                glue_catalog=glue_catalog,
                backend=backend,
                full=args.full,
            )
        )

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
