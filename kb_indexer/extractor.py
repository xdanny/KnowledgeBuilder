from __future__ import annotations

import json
import re
from dataclasses import dataclass
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


@dataclass
class KnowledgeDocument:
    source_path: str
    content: str
    kind: str = "code"

    def s3_key(self, prefix: str) -> str:
        clean_prefix = prefix.rstrip("/")
        normalized_source = self.source_path.replace("\\", "/")
        return f"{clean_prefix}/docs/{normalized_source}.md"


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


def _load_structured(path: Path) -> Any:
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
        cols = ", ".join(info.get("columns", [])[:15]) or "n/a"
        lines.append(
            f"- `{table_name}` | location: `{info.get('location', 'n/a')}` | columns: {cols}"
        )
    return lines


def build_document(
    repo_root: Path,
    file_path: Path,
    glue_catalog: dict[str, dict[str, Any]],
    max_chars_per_doc: int,
) -> KnowledgeDocument:
    relative_path = file_path.relative_to(repo_root).as_posix()
    parsed, raw_text = _load_structured(file_path)
    raw_text = raw_text[:max_chars_per_doc]
    reads, writes = parse_table_refs(raw_text)
    all_tables = sorted(set(reads + writes))
    glue_lines = _render_glue_context(all_tables, glue_catalog)

    if isinstance(parsed, dict):
        structured_preview = json.dumps(parsed, indent=2, default=str)[:6000]
    else:
        structured_preview = raw_text[:3000]

    content = [
        f"# Artifact: {relative_path}",
        "",
        f"- Type: `{file_path.suffix.lower()}`",
        f"- Table reads: {', '.join(reads) if reads else 'none'}",
        f"- Table writes: {', '.join(writes) if writes else 'none'}",
        "",
    ]
    if glue_lines:
        content.append("## Glue Context")
        content.extend(glue_lines)
        content.append("")
    content.append("## Content")
    content.append("```text")
    content.append(structured_preview)
    content.append("```")

    return KnowledgeDocument(source_path=relative_path, content="\n".join(content))


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
                    c.get("Name")
                    for c in table.get("StorageDescriptor", {}).get("Columns", [])
                    if c.get("Name")
                ]
                catalog[full_name] = {
                    "columns": columns,
                    "location": table.get("StorageDescriptor", {}).get("Location"),
                }
    return catalog
