from __future__ import annotations

import json
import math
import sqlite3
import traceback
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from duckduckgo_search import DDGS
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class SearchInput(BaseModel):
    query: str = Field(..., description="Search query.")
    max_results: int = Field(5, ge=1, le=10)


class DuckDuckGoSearchTool(BaseTool):
    name: str = "search"
    description: str = "Search the web for current public information."
    args_schema: type[BaseModel] = SearchInput

    def _fallback(self, query: str) -> List[Dict[str, str]]:
        catalog = {
            "langgraph": {
                "title": "LangGraph",
                "href": "https://langchain-ai.github.io/langgraph/",
                "body": "LangGraph is a framework for building stateful, multi-step agent workflows.",
            },
            "langchain": {
                "title": "LangChain",
                "href": "https://python.langchain.com/",
                "body": "LangChain provides abstractions for building LLM applications and tool-using agents.",
            },
            "duckduckgo": {
                "title": "DuckDuckGo",
                "href": "https://duckduckgo.com/",
                "body": "DuckDuckGo is a privacy-oriented search engine.",
            },
            "sqlite": {
                "title": "SQLite",
                "href": "https://www.sqlite.org/",
                "body": "SQLite is an embedded SQL database engine stored in a local file.",
            },
        }
        lowered = query.lower()
        rows = [value for key, value in catalog.items() if key in lowered]
        return rows or [{"title": "Local fallback", "href": "", "body": f"No live search result; fallback note for query: {query}"}]

    def _run(self, query: str, max_results: int = 5) -> str:
        try:
            with DDGS() as ddgs:
                rows = list(ddgs.text(query, max_results=max_results))
            if not rows:
                rows = self._fallback(query)
        except Exception as exc:
            rows = self._fallback(query)
            return json.dumps(
                {"status": "search_fallback", "reason": repr(exc), "query": query, "results": rows},
                ensure_ascii=False,
            )
        return json.dumps(rows, ensure_ascii=False)


class SqlInput(BaseModel):
    query: str = Field(..., description="Read-only SQLite SQL query.")


class SQLiteDatabaseTool(BaseTool):
    name: str = "sqlite_db"
    description: str = "Run read-only SQL over the local SQLite research database."
    args_schema: type[BaseModel] = SqlInput
    db_path: str

    def _schema_hint(self, query: str, error: str) -> str:
        with sqlite3.connect(self.db_path) as conn:
            tables = pd.read_sql_query(
                "select name from sqlite_master where type='table' order by name",
                conn,
            )["name"].tolist()
            schema = {}
            for table in tables:
                schema[table] = pd.read_sql_query(f"pragma table_info({table})", conn).to_dict(orient="records")
        return json.dumps(
            {"status": "sql_repair_hint", "error": error, "query": query, "schema": schema},
            ensure_ascii=False,
        )

    def _run(self, query: str) -> str:
        query = query.strip()
        if query.startswith("```"):
            query = query.strip("`").replace("sql\n", "", 1).replace("SQL\n", "", 1).strip()
        query = query.rstrip(";")
        if not query.lower().startswith(("select", "with", "pragma")):
            return self._schema_hint(query, "Only read-only SELECT/WITH/PRAGMA statements are allowed.")
        with sqlite3.connect(self.db_path) as conn:
            try:
                df = pd.read_sql_query(query, conn)
            except Exception as exc:
                return self._schema_hint(query, repr(exc))
        return df.to_json(orient="records", force_ascii=False)


class PythonInput(BaseModel):
    code: str = Field(..., description="Python expression or short script. Assign result to variable 'result'.")


class PythonREPLTool(BaseTool):
    name: str = "python_repl"
    description: str = "Execute local Python for numeric analysis. Use only trusted task code."
    args_schema: type[BaseModel] = PythonInput

    def _run(self, code: str) -> str:
        scope: Dict[str, Any] = {"pd": pd, "np": np, "json": json, "math": math}
        try:
            exec(code, {"__builtins__": __builtins__}, scope)
        except Exception as exc:
            try:
                value = eval(code, {"__builtins__": __builtins__, "pd": pd, "np": np, "json": json, "math": math}, scope)
                return repr(value)
            except Exception:
                return json.dumps(
                    {
                        "status": "python_repair_hint",
                        "error": repr(exc),
                        "trace": traceback.format_exc(limit=2),
                        "code": code,
                        "hint": "Assign the final value to variable 'result' or provide a valid Python expression.",
                    },
                    ensure_ascii=False,
                )
        return repr(scope.get("result", None))


class FileInput(BaseModel):
    path: str
    content: str | None = None


class FileSystemTool(BaseTool):
    name: str = "file_system"
    description: str = "Read or write files inside the configured workspace. Omit content to read."
    args_schema: type[BaseModel] = FileInput
    workspace_dir: str

    def _safe_path(self, path: str) -> Path:
        root = Path(self.workspace_dir).resolve()
        target = (root / path).resolve()
        if root not in target.parents and target != root:
            raise ValueError("Path escapes the file workspace.")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def _run(self, path: str, content: str | None = None) -> str:
        target = self._safe_path(path)
        if content is None:
            if not target.exists():
                return json.dumps({"status": "missing_file", "path": path}, ensure_ascii=False)
            return target.read_text(encoding="utf-8")
        target.write_text(content, encoding="utf-8")
        return f"wrote {target}"


def build_tools(sqlite_path: str, workspace_dir: str) -> List[BaseTool]:
    return [
        DuckDuckGoSearchTool(),
        SQLiteDatabaseTool(db_path=sqlite_path),
        PythonREPLTool(),
        FileSystemTool(workspace_dir=workspace_dir),
    ]
