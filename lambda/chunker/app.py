from __future__ import annotations

import json
import os
import re
from typing import Any

import boto3


MAX_CHUNK_CHARS = int(os.environ.get("MAX_CHUNK_CHARS", "1600"))
OVERLAP_CHARS = int(os.environ.get("OVERLAP_CHARS", "220"))
TABLE_PATTERN = re.compile(r"([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)")

s3 = boto3.client("s3")


def split_text(text: str) -> list[str]:
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]

    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + MAX_CHUNK_CHARS, len(text))
        chunk = text[start:end]
        last_break = max(chunk.rfind("\n## "), chunk.rfind("\n\n"))
        if last_break > int(MAX_CHUNK_CHARS * 0.4):
            end = start + last_break
            chunk = text[start:end]
        parts.append(chunk.strip())
        if end == len(text):
            break
        start = max(0, end - OVERLAP_CHARS)
    return [c for c in parts if c]


def parse_table_refs(text: str) -> list[str]:
    return sorted(set(TABLE_PATTERN.findall(text)))


def load_batch(bucket: str, key: str) -> dict[str, Any]:
    response = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read())


def write_batch(bucket: str, key: str, payload: dict[str, Any]) -> None:
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload).encode("utf-8"),
        ContentType="application/json",
    )


def lambda_handler(event: dict[str, Any], _: Any) -> dict[str, Any]:
    bucket = event["bucketName"]
    ingestion_id = event["ingestionJobId"]
    output_files: list[dict[str, Any]] = []

    for input_file in event.get("inputFiles", []):
        for idx, content_batch in enumerate(input_file.get("contentBatches", [])):
            src_key = content_batch["key"]
            payload = load_batch(bucket, src_key)
            file_contents = payload.get("fileContents", [])
            transformed: list[dict[str, Any]] = []
            for file_obj in file_contents:
                content_body = file_obj.get("contentBody", "")
                metadata = file_obj.get("contentMetadata", {})
                chunks = split_text(content_body)
                for chunk_no, chunk in enumerate(chunks, start=1):
                    transformed.append(
                        {
                            "contentBody": chunk,
                            "contentType": file_obj.get("contentType", "TEXT"),
                            "contentMetadata": {
                                **metadata,
                                "chunk_no": str(chunk_no),
                                "table_refs": ",".join(parse_table_refs(chunk)),
                            },
                        }
                    )

            dst_key = (
                f"bedrock-kb-intermediate/{ingestion_id}/chunked/"
                f"{event['knowledgeBaseId']}_{event['dataSourceId']}_{idx}.json"
            )
            write_batch(bucket, dst_key, {"fileContents": transformed})
            output_files.append({"originalFileLocation": input_file["originalFileLocation"], "contentBatches": [{"key": dst_key}]})

    return {"outputFiles": output_files}
