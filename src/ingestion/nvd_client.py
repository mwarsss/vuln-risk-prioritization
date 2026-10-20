"""Pull CVE records from the NVD REST API (base feature vectors)."""
import time
import requests

NVD_BASE_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"


def fetch_cves(api_key: str | None = None, results_per_page: int = 2000, max_pages: int | None = None) -> list[dict]:
    """Paginate through the NVD API, returning raw CVE records."""
    headers = {"apiKey": api_key} if api_key else {}
    sleep_seconds = 0.6 if api_key else 6.0  # respect rate limits (50/30s vs 5/30s)

    records: list[dict] = []
    start_index = 0
    page = 0
    while True:
        resp = requests.get(
            NVD_BASE_URL,
            params={"resultsPerPage": results_per_page, "startIndex": start_index},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        vulns = payload.get("vulnerabilities", [])
        records.extend(v["cve"] for v in vulns if "cve" in v)

        total = payload.get("totalResults", 0)
        start_index += results_per_page
        page += 1
        if start_index >= total or (max_pages and page >= max_pages):
            break
        time.sleep(sleep_seconds)

    return records


def fetch_cve(cve_id: str, api_key: str | None = None, retries: int = 3) -> dict | None:
    """One CVE record from NVD, or None if NVD does not know it."""
    headers = {"apiKey": api_key} if api_key else {}
    for attempt in range(retries):
        resp = requests.get(NVD_BASE_URL, params={"cveId": cve_id}, headers=headers, timeout=30)
        if resp.status_code in (403, 429, 503) and attempt < retries - 1:
            time.sleep(6.0 * (attempt + 1))  # public rate limit is 5 requests / 30 s
            continue
        resp.raise_for_status()
        vulns = resp.json().get("vulnerabilities", [])
        return vulns[0]["cve"] if vulns else None
    return None


_CVSS_FIELDS = {
    "base_severity": "baseSeverity", "attack_vector": "attackVector",
    "attack_complexity": "attackComplexity", "privileges_required": "privilegesRequired",
    "user_interaction": "userInteraction", "scope": "scope",
    "confidentiality_impact": "confidentialityImpact",
    "integrity_impact": "integrityImpact", "availability_impact": "availabilityImpact",
}


def cve_features(cve: dict) -> dict:
    """NVD API 2.0 record -> the publication-time feature row the model trained on.

    CPE data is deliberately not read: it is added during NVD analysis and was
    a label proxy in evaluation (FINDINGS section 11).
    """
    from src.features.prepare_kaggle import _primary_cwe

    desc = " ".join(d["value"] for d in cve.get("descriptions", []) if d.get("lang") == "en")
    cwes = [d["value"] for w in cve.get("weaknesses", []) for d in w.get("description", [])]
    row = {"cve_id": cve["id"], "published_date": cve.get("published"),
           "description": desc, "desc_len": len(desc),
           "primary_cwe": _primary_cwe(cwes), "n_cwe": len(cwes),
           "base_score": None, "exploitability_score": None, "impact_score": None,
           **{k: None for k in _CVSS_FIELDS}}

    metrics = cve.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30"):
        entries = metrics.get(key) or []
        if entries:
            # NVD's own score when it has one, else the CNA's.
            m = next((e for e in entries if e.get("type") == "Primary"), entries[0])
            data = m["cvssData"]
            row.update({"base_score": data.get("baseScore"),
                        "exploitability_score": m.get("exploitabilityScore"),
                        "impact_score": m.get("impactScore"),
                        **{k: data.get(v) for k, v in _CVSS_FIELDS.items()}})
            break
    return row


if __name__ == "__main__":
    cves = fetch_cves(max_pages=1)
    print(f"Fetched {len(cves)} CVE records")
