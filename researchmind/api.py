"""FastAPI interface for ResearchMind.

This is a second front-end (alongside the Streamlit UI in `app.py`) that wraps
the SAME research engine — `run_research_pipeline()` — so any program can drive
the multi-agent pipeline over plain HTTP.

Run it with:

    uvicorn researchmind.api:app --reload

Then open the interactive docs at http://localhost:8000/docs

NOTE: a research run is slow (many LLM calls + the write/critique retry loop),
so `POST /research` blocks until the whole pipeline finishes — expect ~30s-2min
per request. That is fine for this version; a production system would move the
work to a background job and let the client poll for the result.
"""

import os
import sys

# Ensure the project root (the folder containing the `researchmind` package)
# is on sys.path so absolute `researchmind.*` imports resolve regardless of the
# working directory uvicorn is launched from — mirrors app.py.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from researchmind.pipeline import (
    run_research_pipeline,
    report_to_string,
    critic_to_string,
)
from researchmind.database import get_reports, get_report
from researchmind.pdf_utils import create_pdf


app = FastAPI(
    title="ResearchMind API",
    description="REST interface for the multi-agent research pipeline.",
    version="1.0.0",
)

# Allow any origin so a separate frontend (React, etc.) or a quick curl call can
# reach the API during development. Tighten `allow_origins` before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response schemas ────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=1, description="The research topic to investigate.")


class ResearchResponse(BaseModel):
    topic: str
    report: str
    feedback: str
    score: float | None
    attempts: int


class ReportSummary(BaseModel):
    id: int
    topic: str
    report: str
    feedback: str
    created_at: str | None = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row_to_summary(row) -> ReportSummary:
    """Map a DB row (id, topic, report, feedback, created_at) to the API model."""
    return ReportSummary(
        id=row[0],
        topic=row[1],
        report=row[2],
        feedback=row[3],
        created_at=str(row[4]) if len(row) > 4 and row[4] is not None else None,
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest):
    """Run the full multi-agent research pipeline for a topic.

    Blocks until the pipeline finishes, then returns the final report, the
    critic's feedback, the score, and how many write/critique attempts it took.
    The report is also persisted to the history DB and vector store as a side
    effect (same as the Streamlit app).
    """
    topic = req.topic.strip()
    if not topic:
        raise HTTPException(status_code=400, detail="topic must not be empty")

    try:
        state = run_research_pipeline(topic)
    except Exception as exc:  # surface pipeline failures as a clean 500
        raise HTTPException(status_code=500, detail=f"pipeline failed: {exc}")

    return ResearchResponse(
        topic=topic,
        report=report_to_string(state["report"]),
        feedback=critic_to_string(state["feedback"]),
        score=state.get("score"),
        attempts=state.get("attempts", 1),
    )


@app.get("/history", response_model=list[ReportSummary])
def history(limit: int = 20):
    """List past research reports, newest first."""
    rows = get_reports()[:limit]
    return [_row_to_summary(r) for r in rows]


@app.get("/reports/{report_id}", response_model=ReportSummary)
def get_single_report(report_id: int):
    """Fetch one stored report by its id."""
    row = get_report(report_id)
    if row is None:
        raise HTTPException(status_code=404, detail="report not found")
    return _row_to_summary(row)


@app.get("/reports/{report_id}/pdf")
def download_report_pdf(report_id: int):
    """Generate (if needed) and download a stored report as a PDF."""
    row = get_report(report_id)
    if row is None:
        raise HTTPException(status_code=404, detail="report not found")

    topic, report_text = row[1], row[2]
    pdf_path = create_pdf(report_text, topic or f"report_{report_id}")

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=os.path.basename(pdf_path),
    )
