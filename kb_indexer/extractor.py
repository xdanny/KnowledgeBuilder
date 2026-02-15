from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml


TABLE_READ_PATTERNS = [
    re.compile(r"""read(?:\.[a-zA-Z_]+)*\.table\(\s*["']([a-zA-Z0-9_.-]+)["']\s*\)"""),
    re.compile(r"""spark\.table\(\s*["']([a-zA-Z0-9_.-]+)["']\s*\)"""),
    re.compile(r"""(?:from|join)\s+([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)""", re.IGNORECASE),
]

TABLE_WRITE_PATTERNS = [
    re.compile(r"""saveAsTable\(\s*["']([a-zA-Z0-9_.-]+)["']\s*\)"""),
    re.compile(r"""insertInto\(\s*["']([a-zA-Z0-9_.-]+)["']\s*\)"""),
    re.compile(
        r"""(?:insert\s+into|create\s+table(?:\s+if\s+not\s+exists)?)\s+([a-zA-Z0-9_]+\.[a-zA-Z0-9_]+)""",
        re.IGNORECASE,
    ),
]

SQL_METRIC_PATTERNS = [
    re.compile(
        r"""(?P<formula>(?:sum|avg|min|max|count)\s*\([^)]+\))\s+as\s+(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)""",
        re.IGNORECASE,
    )
]


@dataclass
class KnowledgeDocument:
    source_path: str
    content: str
    kind: str = "code"

    def s3_key(self, prefix: str) -> str:
        clean_prefix = prefix.rstrip("/")
        normalized_source = self.source_path.replace("\\", "/")
        return f"{clean_prefix}/docs/{normalized_source}.md"


@dataclass
class MetricEvidence:
    name: str
    formula: str
    source_path: str


@dataclass
class ArtifactAnalysis:
    source_path: str
    file_type: str
    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    metrics: list[MetricEvidence] = field(default_factory=list)
    content_preview: str = ""


def iter_files(repo_root: Path, include_ext: list[str], exclude_dirs: list[str]) -> Iterable[Path]:
    include_set = {ext.lower() for ext in include_ext}
    exclude_set = set(exclude_dirs)
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in exclude_set for part in path.parts):
            continue
        if path.suffix.lower() not in include_set:
            continue
        yield path


def parse_table_refs(text: str) -> tuple[list[str], list[str]]:
    reads: set[str] = set()
    writes: set[str] = set()

    for pattern in TABLE_READ_PATTERNS:
        reads.update(pattern.findall(text))
    for pattern in TABLE_WRITE_PATTERNS:
        writes.update(pattern.findall(text))

    return sorted(reads), sorted(writes)


def parse_metric_refs(text: str, source_path: str) -> list[MetricEvidence]:
    metrics: list[MetricEvidence] = []
    for pattern in SQL_METRIC_PATTERNS:
        for match in pattern.finditer(text):
            formula = match.group("formula").strip()
            name = match.group("name").strip()
            metrics.append(MetricEvidence(name=name, formula=formula, source_path=source_path))
    return metrics


def _load_structured(path: Path) -> tuple[Any, str]:
    suffix = path.suffix.lower()
    raw_text = path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".yaml", ".yml"}:
        try:
            return yaml.safe_load(raw_text), raw_text
        except yaml.YAMLError:
            return None, raw_text
    if suffix == ".json":
        try:
            return json.loads(raw_text), raw_text
        except json.JSONDecodeError:
            return None, raw_text
    return None, raw_text


def _render_glue_context(
    table_names: list[str], glue_catalog: dict[str, dict[str, Any]]
) -> list[str]:
    lines: list[str] = []
    for table_name in table_names:
        info = glue_catalog.get(table_name.lower())
        if not info:
            continue
        cols = ", ".join(info.get("columns", [])[:20]) or "n/a"
        lines.append(
            f"- `{table_name}` | location: `{info.get('location', 'n/a')}` | columns: {cols}"
        )
    return lines


def analyze_artifact(
    repo_root: Path,
    file_path: Path,
    max_chars_per_doc: int,
) -> ArtifactAnalysis:
    relative_path = file_path.relative_to(repo_root).as_posix()
    parsed, raw_text = _load_structured(file_path)
    capped_text = raw_text[:max_chars_per_doc]
    reads, writes = parse_table_refs(capped_text)
    metrics = parse_metric_refs(capped_text, source_path=relative_path)

    if isinstance(parsed, dict):
        preview = json.dumps(parsed, indent=2, default=str)[:7000]
    else:
        preview = capped_text[:3500]

    return ArtifactAnalysis(
        source_path=relative_path,
        file_type=file_path.suffix.lower(),
        reads=reads,
        writes=writes,
        metrics=metrics,
        content_preview=preview,
    )


def build_artifact_document(
    *,
    analysis: ArtifactAnalysis,
    glue_catalog: dict[str, dict[str, Any]],
) -> KnowledgeDocument:
    all_tables = sorted(set(analysis.reads + analysis.writes))
    glue_lines = _render_glue_context(all_tables, glue_catalog)

    lines = [
        f"# Artifact: {analysis.source_path}",
        "",
        "## Metadata",
        f"- Type: `{analysis.file_type}`",
        f"- Table reads: {', '.join(analysis.reads) if analysis.reads else 'none'}",
        f"- Table writes: {', '.join(analysis.writes) if analysis.writes else 'none'}",
        f"- Metrics detected: {', '.join(m.name for m in analysis.metrics) if analysis.metrics else 'none'}",
        "",
    ]
    if glue_lines:
        lines.append("## Glue Context")
        lines.extend(glue_lines)
        lines.append("")
    lines.append("## Content")
    lines.append("```text")
    lines.append(analysis.content_preview)
    lines.append("```")
    return KnowledgeDocument(source_path=analysis.source_path, content="\n".join(lines), kind="artifact")


def build_document(
    repo_root: Path,
    file_path: Path,
    glue_catalog: dict[str, dict[str, Any]],
    max_chars_per_doc: int,
) -> KnowledgeDocument:
    analysis = analyze_artifact(repo_root=repo_root, file_path=file_path, max_chars_per_doc=max_chars_per_doc)
    return build_artifact_document(analysis=analysis, glue_catalog=glue_catalog)


def build_structured_documents(
    *,
    repo_name: str,
    analyses: list[ArtifactAnalysis],
    glue_catalog: dict[str, dict[str, Any]],
    commit_sha: str,
) -> list[KnowledgeDocument]:
    docs: list[KnowledgeDocument] = []
    table_usage: dict[str, dict[str, set[str]]] = {}
    metric_usage: dict[str, MetricEvidence] = {}

    for analysis in analyses:
        for table_name in analysis.reads:
            entry = table_usage.setdefault(table_name.lower(), {"reads": set(), "writes": set()})
            entry["reads"].add(analysis.source_path)
        for table_name in analysis.writes:
            entry = table_usage.setdefault(table_name.lower(), {"reads": set(), "writes": set()})
            entry["writes"].add(analysis.source_path)
        for metric in analysis.metrics:
            metric_usage.setdefault(metric.name.lower(), metric)

    # Table docs
    for table_name, usage in sorted(table_usage.items()):
        glue_info = glue_catalog.get(table_name, {})
        columns = glue_info.get("columns", [])
        confidence = "high" if glue_info else "medium"
        source_of_truth = "glue_catalog" if glue_info else "code_inference"
        table_doc = [
            f"# Table: {table_name}",
            "",
            "## Frontmatter",
            f"- entity_type: table",
            f"- entity_id: {table_name}",
            f"- repo: {repo_name}",
            f"- commit_sha: {commit_sha}",
            f"- source_of_truth: {source_of_truth}",
            f"- confidence: {confidence}",
            "",
            "## Schema",
            f"- Location: `{glue_info.get('location', 'unknown')}`",
            f"- Columns: {', '.join(columns) if columns else 'unknown'}",
            "",
            "## Evidence",
            f"- Read by: {', '.join(sorted(usage['reads'])) if usage['reads'] else 'none'}",
            f"- Written by: {', '.join(sorted(usage['writes'])) if usage['writes'] else 'none'}",
        ]
        docs.append(
            KnowledgeDocument(
                source_path=f"{repo_name}/entities/tables/{table_name}.md",
                content="\n".join(table_doc),
                kind="table",
            )
        )

    # Metric docs
    for metric_name, metric in sorted(metric_usage.items()):
        metric_doc = [
            f"# Metric: {metric_name}",
            "",
            "## Frontmatter",
            "- entity_type: metric",
            f"- entity_id: {metric_name}",
            f"- repo: {repo_name}",
            f"- commit_sha: {commit_sha}",
            "- source_of_truth: code_inference",
            "- confidence: low",
            "",
            "## Definition",
            f"- Formula: `{metric.formula}`",
            "",
            "## Evidence",
            f"- Source file: `{metric.source_path}`",
        ]
        docs.append(
            KnowledgeDocument(
                source_path=f"{repo_name}/entities/metrics/{metric_name}.md",
                content="\n".join(metric_doc),
                kind="metric",
            )
        )

    # Architecture doc
    produced_tables = sorted({table for analysis in analyses for table in analysis.writes})
    consumed_tables = sorted({table for analysis in analyses for table in analysis.reads})
    architecture_doc = [
        f"# Architecture: {repo_name}",
        "",
        "## Frontmatter",
        "- entity_type: architecture",
        f"- entity_id: {repo_name}",
        f"- repo: {repo_name}",
        f"- commit_sha: {commit_sha}",
        "- source_of_truth: generated",
        "- confidence: medium",
        "",
        "## Summary",
        f"- Artifacts analyzed: {len(analyses)}",
        f"- Produced tables: {', '.join(produced_tables) if produced_tables else 'none'}",
        f"- Consumed tables: {', '.join(consumed_tables) if consumed_tables else 'none'}",
        "",
        "## Important Files",
    ]
    important = sorted(
        analyses,
        key=lambda a: len(a.reads) + len(a.writes) + len(a.metrics),
        reverse=True,
    )[:20]
    for analysis in important:
        architecture_doc.append(
            f"- `{analysis.source_path}` | reads={len(analysis.reads)} writes={len(analysis.writes)} metrics={len(analysis.metrics)}"
        )
    docs.append(
        KnowledgeDocument(
            source_path=f"{repo_name}/entities/architecture/overview.md",
            content="\n".join(architecture_doc),
            kind="architecture",
        )
    )
    return docs


def load_glue_catalog(
    glue_client: Any, databases: list[str]
) -> dict[str, dict[str, Any]]:
    if not databases:
        return {}

    catalog: dict[str, dict[str, Any]] = {}
    paginator = glue_client.get_paginator("get_tables")
    for database in databases:
        for page in paginator.paginate(DatabaseName=database):
            for table in page.get("TableList", []):
                db_name = table.get("DatabaseName", database)
                table_name = table.get("Name")
                if not table_name:
                    continue
                full_name = f"{db_name}.{table_name}".lower()
                columns = [
                    column.get("Name")
                    for column in table.get("StorageDescriptor", {}).get("Columns", [])
                    if column.get("Name")
                ]
                catalog[full_name] = {
                    "columns": columns,
                    "location": table.get("StorageDescriptor", {}).get("Location"),
                }
    return catalog
