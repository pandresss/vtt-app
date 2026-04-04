#!/usr/bin/env python3
"""
VTT MCP Server — lets Claude query the transcript database from any conversation.
"""

import sqlite3
import json
import os
from mcp.server.fastmcp import FastMCP

DB_PATH = os.path.expanduser("~/VTT Data/vtt.db")

mcp = FastMCP("VTT Transcripts")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@mcp.tool()
def list_recordings(limit: int = 20) -> str:
    """List recent recordings with metadata (company, role, interviewer, round, date)."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT id, created_at, filename, company, role, interviewer, round, duration_s
               FROM recordings ORDER BY created_at DESC LIMIT ?""",
            (limit,)
        ).fetchall()
    if not rows:
        return "No recordings found."
    results = []
    for r in rows:
        duration = f"{int(r['duration_s'] // 60)}m {int(r['duration_s'] % 60)}s" if r["duration_s"] else "unknown"
        results.append({
            "id": r["id"],
            "date": r["created_at"],
            "company": r["company"] or "unknown",
            "role": r["role"] or "unknown",
            "interviewer": r["interviewer"] or "unknown",
            "round": r["round"] or "unknown",
            "duration": duration,
            "filename": r["filename"],
        })
    return json.dumps(results, indent=2)


@mcp.tool()
def get_transcript(recording_id: int) -> str:
    """Get the full transcript text for a recording by its ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM recordings WHERE id = ?", (recording_id,)
        ).fetchone()
    if not row:
        return f"No recording found with ID {recording_id}."
    return json.dumps({
        "id": row["id"],
        "date": row["created_at"],
        "company": row["company"],
        "role": row["role"],
        "interviewer": row["interviewer"],
        "round": row["round"],
        "topics": row["topics"],
        "duration_s": row["duration_s"],
        "transcript": row["transcript"],
    }, indent=2)


@mcp.tool()
def get_latest_transcript() -> str:
    """Get the most recent transcript."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM recordings ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
    if not row:
        return "No recordings found."
    return json.dumps({
        "id": row["id"],
        "date": row["created_at"],
        "company": row["company"],
        "role": row["role"],
        "interviewer": row["interviewer"],
        "round": row["round"],
        "topics": row["topics"],
        "duration_s": row["duration_s"],
        "transcript": row["transcript"],
    }, indent=2)


@mcp.tool()
def search_recordings(company: str = "", role: str = "", interviewer: str = "") -> str:
    """Search recordings by company, role, or interviewer name (all optional, case-insensitive)."""
    clauses = []
    params = []
    if company:
        clauses.append("LOWER(company) LIKE ?")
        params.append(f"%{company.lower()}%")
    if role:
        clauses.append("LOWER(role) LIKE ?")
        params.append(f"%{role.lower()}%")
    if interviewer:
        clauses.append("LOWER(interviewer) LIKE ?")
        params.append(f"%{interviewer.lower()}%")

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with get_db() as conn:
        rows = conn.execute(
            f"SELECT id, created_at, company, role, interviewer, round, duration_s FROM recordings {where} ORDER BY created_at DESC",
            params
        ).fetchall()
    if not rows:
        return "No matching recordings found."
    results = [dict(r) for r in rows]
    return json.dumps(results, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
