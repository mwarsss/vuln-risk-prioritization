# AI-Assisted Vulnerability Risk Prioritization

Shifting from heuristics (CVSS severity) to predictive probability models for vulnerability triage.

## The Problem: Signal Degradation

- **Data imbalance:** enterprise clouds surface tens of thousands of vulnerabilities; historically <5% are ever actively exploited in the wild.
- **The heuristic trap:** CVSS measures static technical severity, not empirical exploitation risk.
- **High false-positive rate:** flagging thousands of alerts as "Critical" drowns SOC teams in noise and blinds them to the mathematically probable threats.

## The Solution: Applied Machine Learning

- **Feature engineering:** synthesize static CVE metadata with dynamic, real-time threat-intelligence streams. Apply NLP to vulnerability descriptions to surface latent technical indicators.
- **Probabilistic forecasting:** shift the objective from ordinal severity to estimating a continuous exploitation probability, `P(Exploit | Features)`.
- **Model interpretability (XAI):** deploy SHAP values to explain prioritized outputs, producing an actionable, trusted roadmap for DevOps rather than a black-box score.

## What We're Building

| Deliverable | Description |
|---|---|
| Calibrated XGBoost model | Production-ready classifier, continuously updated and validated against CISA KEV positive labels |
| Feature Transformation API | REST interface that parses, encodes, and normalizes raw vulnerability scan output |
| Interactive Inference UI | Svelte app visualizing risk matrices and per-prediction SHAP explanations |
| Synthetic evaluation dataset | Corpus built to stress-test model generalization boundaries before deployment |

## MLOps Pipeline

- **Model training:** Python, scikit-learn, XGBoost — optimized for Precision-Recall AUC (not accuracy) on a heavily imbalanced dataset.
- **Inference engine:** high-throughput FastAPI backend serving containerized ONNX models against incoming scan telemetry.
- **Interactive analytics:** Svelte dashboard exposing risk scores alongside feature-contribution weightings.
- **Deployment:** managed Azure Machine Learning environment with automated model-drift monitoring.

## Empirical ROI

- **Optimize recall** — maximize true-positive identification of statistically probable, weaponized threats, so limited engineering bandwidth goes where it matters.
- **Minimize false positives** — cut the volume of mathematically low-risk alerts that historically trigger unnecessary patch cycles.
- **Quantify risk** — move the security posture from arbitrary ordinal categories to continuous, data-driven probability distributions.

## Beyond the PoC

1. **Online ML** — continuous, automated model retraining within Azure ML as new CVEs emerge.
2. **Graph Networks** — introduce GNNs to model multi-stage attack paths and lateral movement.
3. **LLM Triage** — integrate LLMs to autonomously generate context-aware remediation playbooks.
4. **CI/CD Gates** — dynamic inference gates that block GitHub commits breaching predefined risk-probability thresholds.

See [`DATA_SOURCES.md`](./DATA_SOURCES.md) for the three data sources and how they map to features/labels.
