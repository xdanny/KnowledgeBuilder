from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kb_indexer.extractor import ArtifactAnalysis


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class GraphStore:
    def __init__(self, db_path: str) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                repo TEXT NOT NULL,
                source_path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                last_seen_sha TEXT NOT NULL,
                facts_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (repo, source_path)
            );

            CREATE TABLE IF NOT EXISTS table_refs (
                repo TEXT NOT NULL,
                table_name TEXT NOT NULL,
                source_path TEXT NOT NULL,
                mode TEXT NOT NULL,
                last_seen_sha TEXT NOT NULL,
                PRIMARY KEY (repo, table_name, source_path, mode)
            );

            CREATE TABLE IF NOT EXISTS metric_refs (
                repo TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                source_path TEXT NOT NULL,
                formula TEXT NOT NULL,
                last_seen_sha TEXT NOT NULL,
                PRIMARY KEY (repo, metric_name, source_path)
            );

            CREATE TABLE IF NOT EXISTS symbol_refs (
                repo TEXT NOT NULL,
                symbol_name TEXT NOT NULL,
                symbol_type TEXT NOT NULL,
                source_path TEXT NOT NULL,
                last_seen_sha TEXT NOT NULL,
                PRIMARY KEY (repo, symbol_name, symbol_type, source_path)
            );

            CREATE TABLE IF NOT EXISTS call_edges (
                repo TEXT NOT NULL,
                caller_symbol TEXT NOT NULL,
                callee_symbol TEXT NOT NULL,
                source_path TEXT NOT NULL,
                last_seen_sha TEXT NOT NULL,
                PRIMARY KEY (repo, caller_symbol, callee_symbol, source_path)
            );

            CREATE INDEX IF NOT EXISTS idx_table_refs_repo_table
                ON table_refs (repo, table_name);
            CREATE INDEX IF NOT EXISTS idx_metric_refs_repo_metric
                ON metric_refs (repo, metric_name);
            CREATE INDEX IF NOT EXISTS idx_symbol_refs_repo_source
                ON symbol_refs (repo, source_path);
            CREATE INDEX IF NOT EXISTS idx_call_edges_repo_source
                ON call_edges (repo, source_path);
            """
        )
        self.conn.commit()

    def _facts_from_analysis(self, analysis: ArtifactAnalysis) -> dict[str, Any]:
        return {
            "file_type": analysis.file_type,
            "parser": analysis.parser,
            "reads": analysis.reads,
            "writes": analysis.writes,
            "metrics": [
                {
                    "name": metric.name,
                    "formula": metric.formula,
                    "source_path": metric.source_path,
                }
                for metric in analysis.metrics
            ],
            "symbols": [
                {
                    "name": symbol.name,
                    "symbol_type": symbol.symbol_type,
                    "source_path": symbol.source_path,
                }
                for symbol in analysis.symbols
            ],
            "calls": [
                {
                    "caller": call.caller,
                    "callee": call.callee,
                    "source_path": call.source_path,
                }
                for call in analysis.calls
            ],
        }

    def clear_repo(self, repo: str) -> None:
        self.conn.execute("DELETE FROM files WHERE repo = ?", (repo,))
        self.conn.execute("DELETE FROM table_refs WHERE repo = ?", (repo,))
        self.conn.execute("DELETE FROM metric_refs WHERE repo = ?", (repo,))
        self.conn.execute("DELETE FROM symbol_refs WHERE repo = ?", (repo,))
        self.conn.execute("DELETE FROM call_edges WHERE repo = ?", (repo,))
        self.conn.commit()

    def get_file_hash(self, repo: str, source_path: str) -> str | None:
        row = self.conn.execute(
            "SELECT content_hash FROM files WHERE repo = ? AND source_path = ?",
            (repo, source_path),
        ).fetchone()
        if not row:
            return None
        return str(row["content_hash"])

    def get_file_facts(self, repo: str, source_path: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT facts_json FROM files WHERE repo = ? AND source_path = ?",
            (repo, source_path),
        ).fetchone()
        if not row:
            return None
        try:
            return json.loads(str(row["facts_json"]))
        except json.JSONDecodeError:
            return None

    def _delete_refs_for_source(self, repo: str, source_path: str) -> None:
        self.conn.execute(
            "DELETE FROM table_refs WHERE repo = ? AND source_path = ?",
            (repo, source_path),
        )
        self.conn.execute(
            "DELETE FROM metric_refs WHERE repo = ? AND source_path = ?",
            (repo, source_path),
        )
        self.conn.execute(
            "DELETE FROM symbol_refs WHERE repo = ? AND source_path = ?",
            (repo, source_path),
        )
        self.conn.execute(
            "DELETE FROM call_edges WHERE repo = ? AND source_path = ?",
            (repo, source_path),
        )

    def upsert_file_analysis(
        self,
        *,
        repo: str,
        source_path: str,
        content_hash: str,
        commit_sha: str,
        analysis: ArtifactAnalysis,
    ) -> None:
        self._delete_refs_for_source(repo, source_path)

        for table_name in analysis.reads:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO table_refs (repo, table_name, source_path, mode, last_seen_sha)
                VALUES (?, ?, ?, 'read', ?)
                """,
                (repo, table_name.lower(), source_path, commit_sha),
            )
        for table_name in analysis.writes:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO table_refs (repo, table_name, source_path, mode, last_seen_sha)
                VALUES (?, ?, ?, 'write', ?)
                """,
                (repo, table_name.lower(), source_path, commit_sha),
            )
        for metric in analysis.metrics:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO metric_refs (repo, metric_name, source_path, formula, last_seen_sha)
                VALUES (?, ?, ?, ?, ?)
                """,
                (repo, metric.name.lower(), source_path, metric.formula, commit_sha),
            )
        for symbol in analysis.symbols:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO symbol_refs (repo, symbol_name, symbol_type, source_path, last_seen_sha)
                VALUES (?, ?, ?, ?, ?)
                """,
                (repo, symbol.name, symbol.symbol_type, source_path, commit_sha),
            )
        for call in analysis.calls:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO call_edges (repo, caller_symbol, callee_symbol, source_path, last_seen_sha)
                VALUES (?, ?, ?, ?, ?)
                """,
                (repo, call.caller, call.callee, source_path, commit_sha),
            )

        facts_json = json.dumps(self._facts_from_analysis(analysis), sort_keys=True)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO files (
                repo, source_path, content_hash, last_seen_sha, facts_json, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (repo, source_path, content_hash, commit_sha, facts_json, _now_iso()),
        )
        self.conn.commit()

    def delete_file(self, repo: str, source_path: str) -> dict[str, Any] | None:
        facts = self.get_file_facts(repo, source_path)
        self._delete_refs_for_source(repo, source_path)
        self.conn.execute(
            "DELETE FROM files WHERE repo = ? AND source_path = ?",
            (repo, source_path),
        )
        self.conn.commit()
        return facts

    def get_table_usage(self, repo: str, table_name: str) -> dict[str, list[str]]:
        table = table_name.lower()
        reads = [
            str(row["source_path"])
            for row in self.conn.execute(
                """
                SELECT source_path
                FROM table_refs
                WHERE repo = ? AND table_name = ? AND mode = 'read'
                ORDER BY source_path
                """,
                (repo, table),
            ).fetchall()
        ]
        writes = [
            str(row["source_path"])
            for row in self.conn.execute(
                """
                SELECT source_path
                FROM table_refs
                WHERE repo = ? AND table_name = ? AND mode = 'write'
                ORDER BY source_path
                """,
                (repo, table),
            ).fetchall()
        ]
        return {"reads": reads, "writes": writes}

    def get_metric_definitions(self, repo: str, metric_name: str) -> list[dict[str, str]]:
        metric = metric_name.lower()
        rows = self.conn.execute(
            """
            SELECT formula, source_path
            FROM metric_refs
            WHERE repo = ? AND metric_name = ?
            ORDER BY source_path
            """,
            (repo, metric),
        ).fetchall()
        return [
            {"formula": str(row["formula"]), "source_path": str(row["source_path"])}
            for row in rows
        ]

    def list_tables(self, repo: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT table_name FROM table_refs WHERE repo = ? ORDER BY table_name",
            (repo,),
        ).fetchall()
        return [str(row["table_name"]) for row in rows]

    def list_metrics(self, repo: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT metric_name FROM metric_refs WHERE repo = ? ORDER BY metric_name",
            (repo,),
        ).fetchall()
        return [str(row["metric_name"]) for row in rows]

    def get_repo_file_count(self, repo: str) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) AS c FROM files WHERE repo = ?",
            (repo,),
        ).fetchone()
        return int(row["c"]) if row else 0

    def get_top_files(self, repo: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT
                f.source_path AS source_path,
                COALESCE(r.read_count, 0) AS read_count,
                COALESCE(w.write_count, 0) AS write_count,
                COALESCE(m.metric_count, 0) AS metric_count,
                COALESCE(c.call_count, 0) AS call_count
            FROM files f
            LEFT JOIN (
                SELECT source_path, COUNT(*) AS read_count
                FROM table_refs
                WHERE repo = ? AND mode = 'read'
                GROUP BY source_path
            ) r ON r.source_path = f.source_path
            LEFT JOIN (
                SELECT source_path, COUNT(*) AS write_count
                FROM table_refs
                WHERE repo = ? AND mode = 'write'
                GROUP BY source_path
            ) w ON w.source_path = f.source_path
            LEFT JOIN (
                SELECT source_path, COUNT(*) AS metric_count
                FROM metric_refs
                WHERE repo = ?
                GROUP BY source_path
            ) m ON m.source_path = f.source_path
            LEFT JOIN (
                SELECT source_path, COUNT(*) AS call_count
                FROM call_edges
                WHERE repo = ?
                GROUP BY source_path
            ) c ON c.source_path = f.source_path
            WHERE f.repo = ?
            ORDER BY (
                COALESCE(r.read_count, 0)
                + COALESCE(w.write_count, 0)
                + COALESCE(m.metric_count, 0)
                + COALESCE(c.call_count, 0)
            ) DESC, f.source_path
            LIMIT ?
            """,
            (repo, repo, repo, repo, repo, limit),
        ).fetchall()
        return [
            {
                "source_path": str(row["source_path"]),
                "read_count": int(row["read_count"]),
                "write_count": int(row["write_count"]),
                "metric_count": int(row["metric_count"]),
                "call_count": int(row["call_count"]),
            }
            for row in rows
        ]

    def get_call_edges(self, repo: str, limit: int = 200) -> list[dict[str, str]]:
        rows = self.conn.execute(
            """
            SELECT caller_symbol, callee_symbol, source_path
            FROM call_edges
            WHERE repo = ?
            ORDER BY source_path, caller_symbol, callee_symbol
            LIMIT ?
            """,
            (repo, limit),
        ).fetchall()
        return [
            {
                "caller_symbol": str(row["caller_symbol"]),
                "callee_symbol": str(row["callee_symbol"]),
                "source_path": str(row["source_path"]),
            }
            for row in rows
        ]
