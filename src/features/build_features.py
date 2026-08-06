"""Join NVD + CISA KEV + FIRST EPSS on CVE ID into a training-ready feature table."""
import pandas as pd


def build_feature_table(nvd_records: list[dict], kev_ids: set[str], epss_scores: dict[str, float]) -> pd.DataFrame:
    """Return one row per CVE: structured NVD fields + EPSS baseline + KEV label."""
    rows = []
    for cve in nvd_records:
        cve_id = cve["id"]
        metrics = cve.get("metrics", {})
        cvss = _extract_cvss(metrics)
        rows.append(
            {
                "cve_id": cve_id,
                "description": _extract_description(cve),
                "cvss_base_score": cvss.get("base_score"),
                "attack_vector": cvss.get("attack_vector"),
                "privileges_required": cvss.get("privileges_required"),
                "epss_score": epss_scores.get(cve_id),
                "is_kev": cve_id in kev_ids,  # target label: y = is_kev
            }
        )
    return pd.DataFrame(rows)


def _extract_description(cve: dict) -> str:
    for desc in cve.get("descriptions", []):
        if desc.get("lang") == "en":
            return desc.get("value", "")
    return ""


def _extract_cvss(metrics: dict) -> dict:
    # Prefer CVSS v3.1, fall back to v3.0/v2 — NVD nests these under different keys.
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        entries = metrics.get(key, [])
        if entries:
            data = entries[0]["cvssData"]
            return {
                "base_score": data.get("baseScore"),
                "attack_vector": data.get("attackVector"),
                "privileges_required": data.get("privilegesRequired"),
            }
    return {}
