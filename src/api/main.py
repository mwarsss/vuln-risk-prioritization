"""FastAPI inference endpoint: raw scan telemetry in, calibrated exploit probability + SHAP explanation out."""
import joblib
import shap
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Vulnerability Risk Prioritization API")
model = joblib.load("model.joblib")
explainer = shap.Explainer(model.predict_proba, feature_names=["cvss_base_score", "epss_score"])


class ScanFinding(BaseModel):
    cve_id: str
    cvss_base_score: float | None = None
    epss_score: float | None = None


class RiskResponse(BaseModel):
    cve_id: str
    exploit_probability: float
    shap_contributions: dict[str, float]


@app.post("/score", response_model=RiskResponse)
def score(finding: ScanFinding) -> RiskResponse:
    features = [[finding.cvss_base_score or -1, finding.epss_score or -1]]
    probability = float(model.predict_proba(features)[0][1])
    shap_values = explainer(features)
    contributions = dict(zip(explainer.feature_names, shap_values.values[0][:, 1]))
    return RiskResponse(cve_id=finding.cve_id, exploit_probability=probability, shap_contributions=contributions)
