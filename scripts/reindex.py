#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from kb_indexer.extractor import build_document, iter_files, load_glue_catalog
from kb_indexer.settings import load_settings


@dataclass
class Change:
    status: str
    old_path: str | None
    new_path: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incrementally sync changed repo files to Bedrock KB S3 source and start ingestion."
    )
    parser.add_argument("--config", default="kb_config.yaml", help="Path to config YAML.")
    parser.add_argument(
        "--repo",
        default=".",
        help="Repository path to index.",
    )
    parser.add_argument(
        "--head-sha",
        default="HEAD",
        help="Commit to index (default: HEAD).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Ignore saved state and reindex all tracked files.",
    )
    return parser.parse_args()


def git(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *cmd],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def resolve_sha(repo_root: Path, rev: str) -> str:
    return git(["rev-parse", rev], repo_root)


def commit_exists(repo_root: Path, sha: str) -> bool:
    if not sha:
        return False
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
        cwd=str(repo_root),
        text=True,
        capture_output=True,
    )
    return result.returncode == 0


def list_all_files(repo_root: Path) -> list[str]:
    output = git(["ls-files"], repo_root)
    return [line for line in output.splitlines() if line]


def parse_git_changes(repo_root: Path, base_sha: str, head_sha: str) -> list[Change]:
    output = git(["diff", "--name-status", "--find-renames", base_sha, head_sha], repo_root)
    changes: list[Change] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        status_code = parts[0]
        status = status_code[0]
        if status == "R" and len(parts) >= 3:
            changes.append(Change(status="R", old_path=parts[1], new_path=parts[2]))
        elif status == "D" and len(parts) >= 2:
            changes.append(Change(status="D", old_path=parts[1], new_path=None))
        elif status in {"A", "M"} and len(parts) >= 2:
            changes.append(Change(status=status, old_path=None, new_path=parts[1]))
    return changes


def read_state(s3_client: Any, bucket: str, key: str) -> str | None:
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read().decode("utf-8").strip() or None
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"NoSuchKey", "404"}:
            return None
        raise


def write_state(s3_client: Any, bucket: str, key: str, sha: str) -> None:
    s3_client.put_object(Bucket=bucket, Key=key, Body=sha.encode("utf-8"))


def delete_docs_for_paths(s3_client: Any, bucket: str, prefix: str, paths: list[str]) -> None:
    objects = []
    for path in paths:
        normalized = path.replace("\\", "/")
        objects.append({"Key": f"{prefix.rstrip('/')}/docs/{normalized}.md"})
    if not objects:
        return
    for i in range(0, len(objects), 1000):
        s3_client.delete_objects(Bucket=bucket, Delete={"Objects": objects[i : i + 1000]})


def upload_documents(s3_client: Any, bucket: str, prefix: str, docs: list[Any]) -> None:
    for doc in docs:
        key = doc.s3_key(prefix)
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=doc.content.encode("utf-8"),
            ContentType="text/markdown",
        )


def wait_ingestion_job(client: Any, kb_id: str, ds_id: str, job_id: str) -> None:
    while True:
        response = client.get_ingestion_job(
            knowledgeBaseId=kb_id,
            dataSourceId=ds_id,
            ingestionJobId=job_id,
        )
        status = response["ingestionJob"]["status"]
        if status in {"COMPLETE", "FAILED", "STOPPED"}:
            if status != "COMPLETE":
                raise RuntimeError(f"Ingestion job ended with status: {status}")
            return
        time.sleep(20)


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    repo_root = Path(args.repo).resolve()

    session = boto3.Session(region_name=settings.aws.region)
    s3_client = session.client("s3")
    glue_client = session.client("glue")
    bedrock_agent = session.client("bedrock-agent")

    head_sha = resolve_sha(repo_root, args.head_sha)
    saved_sha = None if args.full else read_state(
        s3_client, settings.aws.source_bucket, settings.aws.state_key
    )

    changed_paths: list[str] = []
    deleted_paths: list[str] = []
    if saved_sha and commit_exists(repo_root, saved_sha):
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
    else:
        changed_paths = list_all_files(repo_root)

    include_ext = settings.indexing.include_extensions
    allowed_paths = {
        p.relative_to(repo_root).as_posix()
        for p in iter_files(repo_root, include_ext, settings.indexing.exclude_dirs)
    }

    changed_filtered = sorted({p for p in changed_paths if p in allowed_paths})
    deleted_filtered = sorted(set(deleted_paths))
    glue_catalog = load_glue_catalog(glue_client, settings.aws.glue_databases)

    docs = []
    for rel_path in changed_filtered:
        file_path = repo_root / rel_path
        if not file_path.exists():
            continue
        docs.append(
            build_document(
                repo_root=repo_root,
                file_path=file_path,
                glue_catalog=glue_catalog,
                max_chars_per_doc=settings.indexing.max_chars_per_doc,
            )
        )

    if deleted_filtered:
        delete_docs_for_paths(
            s3_client, settings.aws.source_bucket, settings.aws.source_prefix, deleted_filtered
        )
    if docs:
        upload_documents(s3_client, settings.aws.source_bucket, settings.aws.source_prefix, docs)

    if (
        settings.indexing.start_ingestion_job
        and settings.aws.knowledge_base_id
        and settings.aws.data_source_id
        and (docs or deleted_filtered)
    ):
        start = bedrock_agent.start_ingestion_job(
            knowledgeBaseId=settings.aws.knowledge_base_id,
            dataSourceId=settings.aws.data_source_id,
            description=f"git incremental index up to {head_sha[:12]}",
        )
        job_id = start["ingestionJob"]["ingestionJobId"]
        if settings.indexing.wait_for_ingestion_job:
            wait_ingestion_job(
                bedrock_agent, settings.aws.knowledge_base_id, settings.aws.data_source_id, job_id
            )

    write_state(s3_client, settings.aws.source_bucket, settings.aws.state_key, head_sha)
    print(
        f"Indexed {len(docs)} files, deleted {len(deleted_filtered)} docs, saved state at {head_sha}."
    )


if __name__ == "__main__":
    main()
