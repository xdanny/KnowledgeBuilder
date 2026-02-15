from __future__ import annotations

import ast
import hashlib
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

SQL_CTE_PATTERN = re.compile(
    r"""\bwith\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\(""",
    re.IGNORECASE,
)

SCALA_SYMBOL_PATTERNS = [
    ("class", re.compile(r"""\bclass\s+([A-Za-z_][A-Za-z0-9_]*)""")),
    ("object", re.compile(r"""\bobject\s+([A-Za-z_][A-Za-z0-9_]*)""")),
    ("trait", re.compile(r"""\btrait\s+([A-Za-z_][A-Za-z0-9_]*)""")),
    ("function", re.compile(r"""\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(""")),
]

SCALA_CALL_PATTERN = re.compile(r"""\b([A-Za-z_][A-Za-z0-9_]*)\s*\(""")
SCALA_CALL_IGNORE = {
    "if",
    "for",
    "while",
    "match",
    "try",
    "catch",
    "new",
    "return",
    "def",
    "class",
    "object",
    "trait",
}


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
class SymbolEvidence:
    name: str
    symbol_type: str
    source_path: str


@dataclass
class CallEvidence:
    caller: str
    callee: str
    source_path: str


@dataclass
class ArtifactAnalysis:
    source_path: str
    file_type: str
    parser: str = "regex"
    reads: list[str] = field(default_factory=list)
    writes: list[str] = field(default_factory=list)
    metrics: list[MetricEvidence] = field(default_factory=list)
    symbols: list[SymbolEvidence] = field(default_factory=list)
    calls: list[CallEvidence] = field(default_factory=list)
    content_preview: str = ""


@dataclass
class StructuredFacts:
    tables: dict[str, dict[str, list[str]]]
    metrics: dict[str, list[dict[str, str]]]
    top_files: list[dict[str, Any]]
    call_edges: list[dict[str, str]]
    file_count: int


def compute_file_hash(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


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
            metrics.append(
                MetricEvidence(
                    name=match.group("name").strip(),
                    formula=match.group("formula").strip(),
                    source_path=source_path,
                )
            )
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


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


class _PythonStructureVisitor(ast.NodeVisitor):
    def __init__(self, source_path: str) -> None:
        self.source_path = source_path
        self.scope_stack: list[str] = []
        self.symbols: list[SymbolEvidence] = []
        self.calls: list[CallEvidence] = []

    def _current_scope(self) -> str:
        if not self.scope_stack:
            return "<module>"
        return ".".join(self.scope_stack)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.symbols.append(
            SymbolEvidence(name=node.name, symbol_type="class", source_path=self.source_path)
        )
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        symbol_type = "method" if self.scope_stack else "function"
        qualname = ".".join([*self.scope_stack, node.name]) if self.scope_stack else node.name
        self.symbols.append(
            SymbolEvidence(name=qualname, symbol_type=symbol_type, source_path=self.source_path)
        )
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.visit_FunctionDef(node)

    def visit_Call(self, node: ast.Call) -> None:
        callee = _call_name(node.func)
        if callee:
            self.calls.append(
                CallEvidence(
                    caller=self._current_scope(),
                    callee=callee,
                    source_path=self.source_path,
                )
            )
        self.generic_visit(node)


def parse_python_structure(text: str, source_path: str) -> tuple[str, list[SymbolEvidence], list[CallEvidence]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return "python_regex_fallback", [], []
    visitor = _PythonStructureVisitor(source_path)
    visitor.visit(tree)
    return "python_ast", visitor.symbols, visitor.calls


def parse_scala_structure(text: str, source_path: str) -> tuple[str, list[SymbolEvidence], list[CallEvidence]]:
    symbols: list[SymbolEvidence] = []
    calls: list[CallEvidence] = []
    for symbol_type, pattern in SCALA_SYMBOL_PATTERNS:
        for match in pattern.finditer(text):
            symbols.append(
                SymbolEvidence(
                    name=match.group(1),
                    symbol_type=symbol_type,
                    source_path=source_path,
                )
            )
    for match in SCALA_CALL_PATTERN.finditer(text):
        callee = match.group(1)
        if callee in SCALA_CALL_IGNORE:
            continue
        calls.append(
            CallEvidence(
                caller="<scala_file>",
                callee=callee,
                source_path=source_path,
            )
        )
    return "scala_regex", symbols, calls


def parse_sql_structure(
    text: str,
    source_path: str,
    reads: list[str],
) -> tuple[str, list[SymbolEvidence], list[CallEvidence]]:
    symbols: list[SymbolEvidence] = []
    calls: list[CallEvidence] = []
    for match in SQL_CTE_PATTERN.finditer(text):
        cte = match.group(1)
        symbols.append(SymbolEvidence(name=cte, symbol_type="cte", source_path=source_path))
    for table_name in reads:
        calls.append(
            CallEvidence(
                caller="<sql_query>",
                callee=table_name,
                source_path=source_path,
            )
        )
    return "sql_regex", symbols, calls


def parse_structure(
    *,
    file_type: str,
    text: str,
    source_path: str,
    reads: list[str],
) -> tuple[str, list[SymbolEvidence], list[CallEvidence]]:
    if file_type == ".py":
        return parse_python_structure(text, source_path=source_path)
    if file_type == ".scala":
        return parse_scala_structure(text, source_path=source_path)
    if file_type == ".sql":
        return parse_sql_structure(text, source_path=source_path, reads=reads)
    return "generic_regex", [], []


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
    parser, symbols, calls = parse_structure(
        file_type=file_path.suffix.lower(),
        text=capped_text,
        source_path=relative_path,
        reads=reads,
    )

    if isinstance(parsed, dict):
        preview = json.dumps(parsed, indent=2, default=str)[:7000]
    else:
        preview = capped_text[:3500]

    return ArtifactAnalysis(
        source_path=relative_path,
        file_type=file_path.suffix.lower(),
        parser=parser,
        reads=reads,
        writes=writes,
        metrics=metrics,
        symbols=symbols,
        calls=calls,
        content_preview=preview,
    )


def build_artifact_document(
    *,
    analysis: ArtifactAnalysis,
    glue_catalog: dict[str, dict[str, Any]],
) -> KnowledgeDocument:
    all_tables = sorted(set(analysis.reads + analysis.writes))
    glue_lines = _render_glue_context(all_tables, glue_catalog)
    symbols_preview = ", ".join(symbol.name for symbol in analysis.symbols[:12]) or "none"
    calls_preview = ", ".join(call.callee for call in analysis.calls[:12]) or "none"

    lines = [
        f"# Artifact: {analysis.source_path}",
        "",
        "## Metadata",
        f"- Type: `{analysis.file_type}`",
        f"- Parser: `{analysis.parser}`",
        f"- Table reads: {', '.join(analysis.reads) if analysis.reads else 'none'}",
        f"- Table writes: {', '.join(analysis.writes) if analysis.writes else 'none'}",
        f"- Metrics detected: {', '.join(metric.name for metric in analysis.metrics) if analysis.metrics else 'none'}",
        f"- Symbols detected: {len(analysis.symbols)} ({symbols_preview})",
        f"- Calls detected: {len(analysis.calls)} ({calls_preview})",
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
    analysis = analyze_artifact(
        repo_root=repo_root,
        file_path=file_path,
        max_chars_per_doc=max_chars_per_doc,
    )
    return build_artifact_document(analysis=analysis, glue_catalog=glue_catalog)


def build_table_document(
    *,
    repo_name: str,
    table_name: str,
    usage: dict[str, list[str]],
    glue_catalog: dict[str, dict[str, Any]],
    commit_sha: str,
) -> KnowledgeDocument:
    glue_info = glue_catalog.get(table_name.lower(), {})
    columns = glue_info.get("columns", [])
    confidence = "high" if glue_info else "medium"
    source_of_truth = "glue_catalog" if glue_info else "code_inference"
    table_doc = [
        f"# Table: {table_name}",
        "",
        "## Frontmatter",
        "- entity_type: table",
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
        f"- Read by: {', '.join(usage.get('reads', [])) if usage.get('reads') else 'none'}",
        f"- Written by: {', '.join(usage.get('writes', [])) if usage.get('writes') else 'none'}",
    ]
    return KnowledgeDocument(
        source_path=f"{repo_name}/entities/tables/{table_name}.md",
        content="\n".join(table_doc),
        kind="table",
    )


def build_metric_document(
    *,
    repo_name: str,
    metric_name: str,
    definitions: list[dict[str, str]],
    commit_sha: str,
) -> KnowledgeDocument:
    confidence = "medium" if len(definitions) == 1 else "low"
    source_of_truth = "code_inference"
    definition_lines = []
    for definition in definitions[:20]:
        definition_lines.append(
            f"- `{definition.get('formula', '')}` from `{definition.get('source_path', '')}`"
        )
    if not definition_lines:
        definition_lines.append("- unknown")
    metric_doc = [
        f"# Metric: {metric_name}",
        "",
        "## Frontmatter",
        "- entity_type: metric",
        f"- entity_id: {metric_name}",
        f"- repo: {repo_name}",
        f"- commit_sha: {commit_sha}",
        f"- source_of_truth: {source_of_truth}",
        f"- confidence: {confidence}",
        "",
        "## Definitions",
        *definition_lines,
    ]
    return KnowledgeDocument(
        source_path=f"{repo_name}/entities/metrics/{metric_name}.md",
        content="\n".join(metric_doc),
        kind="metric",
    )


def build_architecture_document(
    *,
    repo_name: str,
    commit_sha: str,
    facts: StructuredFacts,
) -> KnowledgeDocument:
    produced_tables = sorted(
        table for table, usage in facts.tables.items() if usage.get("writes")
    )
    consumed_tables = sorted(
        table for table, usage in facts.tables.items() if usage.get("reads")
    )
    lines = [
        f"# Architecture: {repo_name}",
        "",
        "## Frontmatter",
        "- entity_type: architecture",
        f"- entity_id: {repo_name}",
        f"- repo: {repo_name}",
        f"- commit_sha: {commit_sha}",
        "- source_of_truth: graph_store",
        "- confidence: medium",
        "",
        "## Summary",
        f"- Files tracked: {facts.file_count}",
        f"- Produced tables: {', '.join(produced_tables) if produced_tables else 'none'}",
        f"- Consumed tables: {', '.join(consumed_tables) if consumed_tables else 'none'}",
        f"- Metrics tracked: {', '.join(sorted(facts.metrics.keys())) if facts.metrics else 'none'}",
        "",
        "## Impacted Files (Top)",
    ]
    for row in facts.top_files:
        lines.append(
            f"- `{row.get('source_path')}` | reads={row.get('read_count')} writes={row.get('write_count')} metrics={row.get('metric_count')} calls={row.get('call_count')}"
        )
    if not facts.top_files:
        lines.append("- none")
    lines.extend(["", "## Call Graph Edges (Sample)"])
    for edge in facts.call_edges[:40]:
        lines.append(
            f"- `{edge.get('caller_symbol')}` -> `{edge.get('callee_symbol')}` ({edge.get('source_path')})"
        )
    if not facts.call_edges:
        lines.append("- none")

    return KnowledgeDocument(
        source_path=f"{repo_name}/entities/architecture/overview.md",
        content="\n".join(lines),
        kind="architecture",
    )


def build_structured_documents(
    *,
    repo_name: str,
    analyses: list[ArtifactAnalysis],
    glue_catalog: dict[str, dict[str, Any]],
    commit_sha: str,
) -> list[KnowledgeDocument]:
    # Backward-compatible full rebuild from in-memory analyses.
    tables: dict[str, dict[str, list[str]]] = {}
    metrics: dict[str, list[dict[str, str]]] = {}
    for analysis in analyses:
        for table_name in analysis.reads:
            usage = tables.setdefault(table_name.lower(), {"reads": [], "writes": []})
            usage["reads"].append(analysis.source_path)
        for table_name in analysis.writes:
            usage = tables.setdefault(table_name.lower(), {"reads": [], "writes": []})
            usage["writes"].append(analysis.source_path)
        for metric in analysis.metrics:
            defs = metrics.setdefault(metric.name.lower(), [])
            defs.append({"formula": metric.formula, "source_path": metric.source_path})

    docs: list[KnowledgeDocument] = []
    for table_name in sorted(tables.keys()):
        docs.append(
            build_table_document(
                repo_name=repo_name,
                table_name=table_name,
                usage=tables[table_name],
                glue_catalog=glue_catalog,
                commit_sha=commit_sha,
            )
        )
    for metric_name in sorted(metrics.keys()):
        docs.append(
            build_metric_document(
                repo_name=repo_name,
                metric_name=metric_name,
                definitions=metrics[metric_name],
                commit_sha=commit_sha,
            )
        )
    docs.append(
        build_architecture_document(
            repo_name=repo_name,
            commit_sha=commit_sha,
            facts=StructuredFacts(
                tables=tables,
                metrics=metrics,
                top_files=sorted(
                    [
                        {
                            "source_path": analysis.source_path,
                            "read_count": len(analysis.reads),
                            "write_count": len(analysis.writes),
                            "metric_count": len(analysis.metrics),
                            "call_count": len(analysis.calls),
                        }
                        for analysis in analyses
                    ],
                    key=lambda row: row["read_count"] + row["write_count"] + row["metric_count"],
                    reverse=True,
                )[:20],
                call_edges=[
                    {
                        "caller_symbol": call.caller,
                        "callee_symbol": call.callee,
                        "source_path": call.source_path,
                    }
                    for analysis in analyses
                    for call in analysis.calls
                ],
                file_count=len(analyses),
            ),
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
