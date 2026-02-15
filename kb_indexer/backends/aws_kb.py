from __future__ import annotations

import time
from typing import Any

from kb_indexer.backends.base import IndexBackend
from kb_indexer.extractor import KnowledgeDocument
from kb_indexer.settings import AppSettings


class AwsKbBackend(IndexBackend):
    def __init__(self, s3_client: Any, bedrock_agent_client: Any, settings: AppSettings) -> None:
        if not settings.aws.source_bucket:
            raise ValueError("aws.source_bucket is required for aws_kb backend.")
        self.s3 = s3_client
        self.bedrock_agent = bedrock_agent_client
        self.settings = settings
        self.bucket = settings.aws.source_bucket
        self.prefix = settings.aws.source_prefix
        self.docs_uploaded = 0
        self.docs_deleted = 0
        self.ingestion_job_id: str | None = None

    def upsert_documents(self, docs: list[KnowledgeDocument]) -> None:
        for doc in docs:
            key = doc.s3_key(self.prefix)
            self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=doc.content.encode("utf-8"),
                ContentType="text/markdown",
            )
        self.docs_uploaded += len(docs)

    def delete_documents(self, source_paths: list[str]) -> None:
        objects = []
        for path in source_paths:
            normalized = path.replace("\\", "/")
            objects.append({"Key": f"{self.prefix.rstrip('/')}/docs/{normalized}.md"})
        if not objects:
            return
        for i in range(0, len(objects), 1000):
            self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": objects[i : i + 1000]})
        self.docs_deleted += len(objects)

    def delete_by_prefix(self, prefixes: list[str]) -> None:
        for prefix in prefixes:
            normalized = prefix.replace("\\", "/").rstrip("/")
            s3_prefix = f"{self.prefix.rstrip('/')}/docs/{normalized}"
            continuation_token = None
            while True:
                kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": s3_prefix}
                if continuation_token:
                    kwargs["ContinuationToken"] = continuation_token
                response = self.s3.list_objects_v2(**kwargs)
                objects = [{"Key": obj["Key"]} for obj in response.get("Contents", [])]
                if objects:
                    for i in range(0, len(objects), 1000):
                        self.s3.delete_objects(
                            Bucket=self.bucket,
                            Delete={"Objects": objects[i : i + 1000]},
                        )
                    self.docs_deleted += len(objects)
                if not response.get("IsTruncated"):
                    break
                continuation_token = response.get("NextContinuationToken")

    def _wait_ingestion_job(self, kb_id: str, ds_id: str, job_id: str) -> None:
        while True:
            response = self.bedrock_agent.get_ingestion_job(
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

    def _start_ingestion_if_needed(self) -> None:
        kb_id = self.settings.aws.knowledge_base_id
        ds_id = self.settings.aws.data_source_id
        if not kb_id or not ds_id:
            return
        if not self.settings.backend.aws_kb.start_ingestion_job:
            return
        if self.docs_uploaded == 0 and self.docs_deleted == 0:
            return

        response = self.bedrock_agent.start_ingestion_job(
            knowledgeBaseId=kb_id,
            dataSourceId=ds_id,
            description="incremental kb sync",
        )
        self.ingestion_job_id = response["ingestionJob"]["ingestionJobId"]
        if self.settings.backend.aws_kb.wait_for_ingestion_job:
            self._wait_ingestion_job(kb_id, ds_id, self.ingestion_job_id)

    def finalize(self) -> dict[str, Any]:
        self._start_ingestion_if_needed()
        return {
            "backend": "aws_kb",
            "uploaded_docs": self.docs_uploaded,
            "deleted_docs": self.docs_deleted,
            "ingestion_job_id": self.ingestion_job_id,
        }
