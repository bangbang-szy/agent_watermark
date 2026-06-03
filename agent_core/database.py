from __future__ import annotations

import sqlite3
from pathlib import Path


def init_demo_database(path: str) -> None:
    """Create a small but real SQLite database for retrieval and chaining tasks."""
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute("create table if not exists papers(id integer primary key, title text, year integer, citations integer, area text)")
        cur.execute("create table if not exists agents(id integer primary key, name text, framework text, tools integer, success_rate real)")
        cur.execute("delete from papers")
        cur.execute("delete from agents")
        cur.executemany(
            "insert into papers(title, year, citations, area) values(?,?,?,?)",
            [
                ("Behavioral Watermarks for Tool Agents", 2025, 42, "watermarking"),
                ("Robust Decoding from Execution Logs", 2024, 31, "observability"),
                ("LangGraph Control Flow for Agents", 2024, 75, "agents"),
                ("Multi Statistic Voting Codes", 2023, 54, "statistics"),
            ],
        )
        cur.executemany(
            "insert into agents(name, framework, tools, success_rate) values(?,?,?,?)",
            [
                ("react-sql", "langchain", 4, 0.81),
                ("graph-researcher", "langgraph", 5, 0.88),
                ("toolformer-lite", "custom", 3, 0.74),
            ],
        )
        conn.commit()
