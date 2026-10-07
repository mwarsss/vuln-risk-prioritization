"""Serve the gated triage policy from FINDINGS section 9.

EPSS (read as of publication + 1 day, never a current pull) orders the head of
the queue; the content model orders the tail. Two endpoints:

  POST /score  one CVE: content score, percentile, EPSS tier, explanation
  POST /rank   a backlog: the gated queue order, plus whether the budget is deep
               enough for the gate to beat EPSS alone

The model is loaded lazily from $VRP_MODEL (default ./model.joblib), produced by
`python -m src.model.train`.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model.bundle import MIN_USEFUL_BUDGET, ContentBundle
from src.model.evaluate import CAT_COLS, NUM_COLS

app = FastAPI(title="Vulnerability Risk Prioritization API")


@lru_cache(maxsize=1)
def get_bundle() -> ContentBundle:
    path = os.environ.get("VRP_MODEL", "model.joblib")
    if not os.path.exists(path):
        raise HTTPException(503, f"model not found at {path!r}; run `python -m src.model.train`")
    return ContentBundle.load(path)


class Finding(BaseModel):
    """NVD fields as published. Anything unknown is left null, exactly as in training."""

    cve_id: str
    description: str = ""
    epss_score: float | None = Field(
        None, ge=0, le=1,
        description="EPSS as of publication + 1 day. A current EPSS pull leaks outcome for old CVEs.",
    )
    base_score: float | None = None
    exploitability_score: float | None = None
    impact_score: float | None = None
    n_cwe: int | None = None
    n_vendors: int | None = None
    n_cpe: int | None = None
    base_severity: str | None = None
    attack_vector: str | None = None
    attack_complexity: str | None = None
    privileges_required: str | None = None
    user_interaction: str | None = None
    scope: str | None = None
    confidentiality_impact: str | None = None
    integrity_impact: str | None = None
    availability_impact: str | None = None
    primary_cwe: str | None = None
    vendor: str | None = None


class ScoreResponse(BaseModel):
    cve_id: str
    content_score: float
    content_percentile: float
    epss_tier: str
    contributions: dict[str, float]


class RankRequest(BaseModel):
    findings: list[Finding] = Field(min_length=1, max_length=50_000)
    budget: int = Field(gt=0, description="How many CVEs the team can patch")


class RankResponse(BaseModel):
    order: list[str]  # cve_ids to patch, first = most urgent
    selected: list[str]  # order[:budget]
    gate_helps: bool
    note: str


def _frame(findings: list[Finding]) -> pd.DataFrame:
    df = pd.DataFrame([f.model_dump() for f in findings])
    df["desc_len"] = df["description"].str.len()
    return df[["cve_id", "description", "epss_score", "desc_len", *CAT_COLS,
               *[c for c in NUM_COLS if c != "desc_len"]]]


def _tier(epss: float | None, cutoff: float) -> str:
    if epss is None:
        return "unscored"  # ~8% of CVEs one day after publication have no EPSS yet
    return "head" if epss >= cutoff else "tail"


@app.post("/score", response_model=ScoreResponse)
def score(finding: Finding) -> ScoreResponse:
    bundle = get_bundle()
    df = _frame([finding])
    s = bundle.score(df)
    return ScoreResponse(
        cve_id=finding.cve_id,
        content_score=float(s[0]),
        content_percentile=float(bundle.percentile(s)[0]),
        epss_tier=_tier(finding.epss_score, bundle.gate_cutoff),
        contributions=bundle.explain(df)[0],
    )


@app.post("/rank", response_model=RankResponse)
def rank(req: RankRequest) -> RankResponse:
    bundle = get_bundle()
    df = _frame(req.findings)
    epss = df["epss_score"].astype(float).fillna(-1.0).to_numpy()
    key = bundle.rank_key(df, epss)
    order = df["cve_id"].to_numpy()[np.argsort(-key, kind="stable")].tolist()
    helps = req.budget >= MIN_USEFUL_BUDGET
    note = (
        "Gated order (EPSS head, content tail)."
        if helps else
        f"Budget below ~{MIN_USEFUL_BUDGET}: FINDINGS section 9 shows the gate is worse than "
        "EPSS alone here. Taking the head by EPSS only."
    )
    if not helps:
        order = df["cve_id"].to_numpy()[np.argsort(-epss, kind="stable")].tolist()
    return RankResponse(order=order, selected=order[: req.budget], gate_helps=helps, note=note)
