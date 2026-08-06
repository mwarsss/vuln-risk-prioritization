# Data Sources

Three sources, each with a distinct role in the pipeline: features, ground-truth labels, and probability baselines.

## 1. NVD — National Vulnerability Database (Features)

Provides the base feature vectors for every CVE: CVSS vectors, CWE categories, affected product/vendor (CPE), and free-text vulnerability descriptions. Apply NLP to the descriptions to surface latent technical indicators (attack vector, privileges required, etc.) not captured by the structured fields alone.

- **API:** `https://services.nvd.nist.gov/rest/json/cves/2.0`
- **Auth:** optional API key (raises rate limit from 5 req/30s to 50 req/30s) — request at https://nvd.nist.gov/developers/request-an-api-key
- **Format:** JSON, paginated (`resultsPerPage`, `startIndex`)

## 2. CISA KEV — Known Exploited Vulnerabilities Catalog (Ground Truth)

The empirical positive class. CISA KEV lists only vulnerabilities with **verified, active exploitation** — used strictly to label training data (1 = actively weaponized, appears in KEV; 0 = not yet observed exploited). This is what makes the target variable "exploitation probability" rather than a heuristic severity score.

- **Feed:** `https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json`
- **Auth:** none, public feed
- **Format:** JSON, flat list of CVE entries with `dateAdded`, `requiredAction`, `dueDate`

## 3. FIRST EPSS — Exploit Prediction Scoring System (Baselines)

Supplies pre-trained exploitation-probability estimates per CVE, updated daily. Used as a **baseline to calibrate against**, not as the model itself — the project's XGBoost model should ideally exceed or explain EPSS's opaque score. Also usable as a time-series feature (score trend over the last N days) to inform a custom 30-day exploitation forecast.

- **API:** `https://api.first.org/data/v1/epss?cve=CVE-XXXX-XXXXX`
- **Bulk daily CSV:** `https://epss.cyentia.com/epss_scores-YYYY-MM-DD.csv.gz`
- **Auth:** none, public

## Join key

All three sources join on **CVE ID** (`CVE-YYYY-NNNNN`). Pipeline: pull NVD records → label with CISA KEV membership → attach EPSS score/history as a baseline feature → train XGBoost on the combined feature set with KEV membership (or an EPSS-informed continuous target) as `y`.
