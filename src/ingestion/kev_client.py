"""Pull the CISA KEV catalog (empirical ground-truth positive class)."""
import requests

KEV_FEED_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"


def fetch_kev_cve_ids() -> set[str]:
    """Return the set of CVE IDs with verified, active exploitation."""
    resp = requests.get(KEV_FEED_URL, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    return {entry["cveID"] for entry in payload.get("vulnerabilities", [])}


if __name__ == "__main__":
    kev_ids = fetch_kev_cve_ids()
    print(f"KEV catalog: {len(kev_ids)} actively exploited CVEs")
