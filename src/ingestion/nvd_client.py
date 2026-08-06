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


if __name__ == "__main__":
    cves = fetch_cves(max_pages=1)
    print(f"Fetched {len(cves)} CVE records")
