"""FastAPI endpoint: CVE IDs in, banded patch ranking out.

    uvicorn src.api.main:app
    curl -X POST localhost:8000/rank -H 'content-type: application/json' \\
         -d '{"cve_ids": ["CVE-2024-3400", "CVE-2023-4966"]}'

Replaces the scaffold, which loaded a CVSS+EPSS model trained on a random split
— the leak pattern this project spent sections 1-6 removing.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.rank import load_bundle, rank_cves

app = FastAPI(title="Vulnerability Risk Prioritization API")
bundle = load_bundle()
MAX_IDS = 200  # NVD is fetched per CVE; larger scans belong in the CLI


class RankRequest(BaseModel):
    cve_ids: list[str] = Field(min_length=1)


class RankedCVE(BaseModel):
    rank: int
    cve_id: str
    band: str
    epss: float | None
    model_score: float | None
    base_score: float | None
    note: str


@app.get("/health")
def health() -> dict:
    return {"trained_through": bundle["trained_through"], "n_train": bundle["n_train"]}


@app.post("/rank", response_model=list[RankedCVE])
def rank(req: RankRequest) -> list[RankedCVE]:
    if len(req.cve_ids) > MAX_IDS:
        raise HTTPException(413, f"at most {MAX_IDS} CVE IDs per request; use `python -m src.rank`")
    df = rank_cves(req.cve_ids, bundle, verbose=False)
    cols = ["rank", "cve_id", "band", "epss", "model_score", "base_score", "note"]
    records = df[cols].astype(object).where(df[cols].notna(), None).to_dict("records")
    return [RankedCVE(**r) for r in records]
