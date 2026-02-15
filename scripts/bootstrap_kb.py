#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from typing import Any

import boto3
from botocore.exceptions import ClientError

from kb_indexer.settings import AppSettings, load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Bedrock Knowledge Base + S3 Vector index + S3 data source."
    )
    parser.add_argument("--config", default="kb_config.yaml", help="Path to config YAML.")
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait until the knowledge base becomes ACTIVE.",
    )
    return parser.parse_args()


def ensure_bucket(s3_client: Any, bucket: str, region: str) -> None:
    try:
        s3_client.head_bucket(Bucket=bucket)
        return
    except ClientError:
        pass

    params: dict[str, Any] = {"Bucket": bucket}
    if region != "us-east-1":
        params["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3_client.create_bucket(**params)


def find_kb_by_name(client: Any, kb_name: str) -> str | None:
    paginator = client.get_paginator("list_knowledge_bases")
    for page in paginator.paginate():
        for summary in page.get("knowledgeBaseSummaries", []):
            if summary.get("name") == kb_name:
                return summary.get("knowledgeBaseId")
    return None


def find_ds_by_name(client: Any, knowledge_base_id: str, data_source_name: str) -> str | None:
    paginator = client.get_paginator("list_data_sources")
    for page in paginator.paginate(knowledgeBaseId=knowledge_base_id):
        for summary in page.get("dataSourceSummaries", []):
            if summary.get("name") == data_source_name:
                return summary.get("dataSourceId")
    return None


def ensure_vector_bucket(s3vectors_client: Any, vector_bucket_name: str) -> str:
    try:
        response = s3vectors_client.create_vector_bucket(vectorBucketName=vector_bucket_name)
        return response["vectorBucketArn"]
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ConflictException":
            raise
        response = s3vectors_client.get_vector_bucket(vectorBucketName=vector_bucket_name)
        return response["vectorBucketArn"]


def ensure_vector_index(
    s3vectors_client: Any,
    vector_bucket_name: str,
    vector_index_name: str,
    dimension: int,
    distance_metric: str,
) -> str:
    try:
        response = s3vectors_client.create_index(
            vectorBucketName=vector_bucket_name,
            indexName=vector_index_name,
            dataType="float32",
            dimension=dimension,
            distanceMetric=distance_metric,
        )
        return response["indexArn"]
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ConflictException":
            raise
        response = s3vectors_client.get_index(
            vectorBucketName=vector_bucket_name,
            indexName=vector_index_name,
        )
        return response["indexArn"]


def build_ingestion_config(settings: AppSettings) -> dict[str, Any]:
    config: dict[str, Any] = {
        "chunkingConfiguration": {
            "chunkingStrategy": "FIXED_SIZE",
            "fixedSizeChunkingConfiguration": {
                "maxTokens": 400,
                "overlapPercentage": 20,
            },
        }
    }

    chunk_lambda_arn = settings.bootstrap.chunk_lambda_arn
    intermediate_s3_uri = settings.bootstrap.intermediate_s3_uri
    if chunk_lambda_arn and intermediate_s3_uri:
        config["customTransformationConfiguration"] = {
            "intermediateStorage": {"s3Location": {"uri": intermediate_s3_uri}},
            "transformations": [
                {
                    "stepToApply": "POST_CHUNKING",
                    "transformationFunction": {
                        "transformationLambdaConfiguration": {
                            "lambdaArn": chunk_lambda_arn,
                        }
                    },
                }
            ],
        }

    context_model_arn = settings.bootstrap.context_enrichment_model_arn
    if context_model_arn:
        config["contextEnrichmentConfiguration"] = {
            "type": "BEDROCK_FOUNDATION_MODEL",
            "bedrockFoundationModelConfiguration": {
                "modelArn": context_model_arn,
                "enrichmentStrategyConfiguration": {
                    "method": "CHUNK_ENTITY_EXTRACTION",
                },
            },
        }
    return config


def wait_for_active_kb(client: Any, knowledge_base_id: str) -> None:
    for _ in range(60):
        response = client.get_knowledge_base(knowledgeBaseId=knowledge_base_id)
        status = response["knowledgeBase"]["status"]
        if status == "ACTIVE":
            return
        if status in {"FAILED", "DELETE_UNSUCCESSFUL"}:
            raise RuntimeError(f"Knowledge base status is {status}")
        time.sleep(10)
    raise TimeoutError("Knowledge base did not become ACTIVE within 10 minutes.")


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    region = settings.aws.region

    if not settings.bootstrap.bedrock_role_arn:
        raise ValueError("bootstrap.bedrock_role_arn is required.")
    if not settings.bootstrap.embedding_model_arn:
        raise ValueError("bootstrap.embedding_model_arn is required.")
    if not settings.aws.source_bucket:
        raise ValueError("aws.source_bucket is required.")
    if not settings.bootstrap.knowledge_base_name:
        raise ValueError("bootstrap.knowledge_base_name is required.")
    if not settings.bootstrap.data_source_name:
        raise ValueError("bootstrap.data_source_name is required.")
    if not settings.bootstrap.vector_bucket_name:
        raise ValueError("bootstrap.vector_bucket_name is required.")
    if not settings.bootstrap.vector_index_name:
        raise ValueError("bootstrap.vector_index_name is required.")

    session = boto3.Session(region_name=region)
    s3_client = session.client("s3")
    s3vectors_client = session.client("s3vectors")
    bedrock_agent = session.client("bedrock-agent")

    ensure_bucket(s3_client, settings.aws.source_bucket, region)
    vector_bucket_arn = ensure_vector_bucket(s3vectors_client, settings.bootstrap.vector_bucket_name)
    index_arn = ensure_vector_index(
        s3vectors_client,
        vector_bucket_name=settings.bootstrap.vector_bucket_name,
        vector_index_name=settings.bootstrap.vector_index_name,
        dimension=settings.bootstrap.vector_dimension,
        distance_metric=settings.bootstrap.vector_distance_metric,
    )

    knowledge_base_id = find_kb_by_name(
        bedrock_agent, settings.bootstrap.knowledge_base_name
    )
    if knowledge_base_id is None:
        response = bedrock_agent.create_knowledge_base(
            name=settings.bootstrap.knowledge_base_name,
            roleArn=settings.bootstrap.bedrock_role_arn,
            knowledgeBaseConfiguration={
                "type": "VECTOR",
                "vectorKnowledgeBaseConfiguration": {
                    "embeddingModelArn": settings.bootstrap.embedding_model_arn
                },
            },
            storageConfiguration={
                "type": "S3_VECTORS",
                "s3VectorsConfiguration": {
                    "vectorBucketArn": vector_bucket_arn,
                    "indexArn": index_arn,
                    "indexName": settings.bootstrap.vector_index_name,
                },
            },
        )
        knowledge_base_id = response["knowledgeBase"]["knowledgeBaseId"]

    if args.wait:
        wait_for_active_kb(bedrock_agent, knowledge_base_id)

    data_source_id = find_ds_by_name(
        bedrock_agent,
        knowledge_base_id=knowledge_base_id,
        data_source_name=settings.bootstrap.data_source_name,
    )
    if data_source_id is None:
        response = bedrock_agent.create_data_source(
            knowledgeBaseId=knowledge_base_id,
            name=settings.bootstrap.data_source_name,
            dataSourceConfiguration={
                "type": "S3",
                "s3Configuration": {
                    "bucketArn": f"arn:aws:s3:::{settings.aws.source_bucket}",
                    "inclusionPrefixes": [settings.aws.source_prefix.rstrip("/") + "/docs/"],
                },
            },
            vectorIngestionConfiguration=build_ingestion_config(settings),
        )
        data_source_id = response["dataSource"]["dataSourceId"]

    print(
        json.dumps(
            {
                "knowledge_base_id": knowledge_base_id,
                "data_source_id": data_source_id,
                "vector_bucket_name": settings.bootstrap.vector_bucket_name,
                "vector_index_name": settings.bootstrap.vector_index_name,
                "source_bucket": settings.aws.source_bucket,
                "source_prefix": settings.aws.source_prefix,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
