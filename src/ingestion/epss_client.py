"""Pull FIRST EPSS exploitation-probability baselines."""
import requests

EPSS_API_URL = "https://api.first.org/data/v1/epss"


def fetch_epss_scores(cve_ids: list[str]) -> dict[str, float]:
    """Return {cve_id: epss_score} for the given CVE IDs, batched at 100 per request."""
    scores: dict[str, float] = {}
    for i in range(0, len(cve_ids), 100):
        batch = cve_ids[i : i + 100]
        resp = requests.get(EPSS_API_URL, params={"cve": ",".join(batch)}, timeout=30)
        resp.raise_for_status()
        for row in resp.json().get("data", []):
            scores[row["cve"]] = float(row["epss"])
    return scores


if __name__ == "__main__":
    print(fetch_epss_scores(["CVE-2021-44228"]))  # Log4Shell, sanity check
