"""Pull the CISA KEV catalog (empirical ground-truth positive class).

The catalog carries a `dateAdded` per entry, which the Kaggle bulk mirror drops.
Without it the only expressible target is "appears in KEV as of whenever the data
was pulled" — an open-ended label that counts a CVE exploited three years after
publication as something the model should have caught on day one, and that
silently shifts every time CISA backfills the catalogue.

With `dateAdded` the target can be bounded to a horizon
(`src.schema.build_horizon_label`), which is the question a triage team actually
faces: given this CVE now, will it be weaponised inside the next N days?
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

log = logging.getLogger(__name__)

KEV_FEED_URL = "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"
CACHE_PATH = Path("data/raw/kev_catalog.parquet")


def fetch_kev_catalog(*, cache_path: Path | None = CACHE_PATH,
                      refresh: bool = False, timeout: int = 30) -> pd.DataFrame:
    """Return one row per KEV entry: cve_id, kev_date_added, vendor/product, ransomware flag.

    Unlike the EPSS archive, this feed is a *current* snapshot — CISA publishes
    only the live catalogue, not dated historical versions. `kev_date_added` is
    what makes point-in-time reconstruction possible anyway: an entry added
    2025-03-01 was simply absent from the catalogue before that date.
    """
    if cache_path is not None and cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    log.info("fetching CISA KEV catalog")
    resp = requests.get(KEV_FEED_URL, timeout=timeout)
    resp.raise_for_status()
    entries = resp.json().get("vulnerabilities", [])

    df = pd.DataFrame(
        [
            {
                "cve_id": e["cveID"],
                "kev_date_added": e.get("dateAdded"),
                "kev_vendor": e.get("vendorProject"),
                "kev_product": e.get("product"),
                # Post-hoc attribute: true only once ransomware use is observed.
                # Never usable as a feature, retained for cohort analysis.
                "kev_ransomware": e.get("knownRansomwareCampaignUse", "").lower() == "known",
            }
            for e in entries
        ]
    )
    df["kev_date_added"] = pd.to_datetime(df["kev_date_added"], errors="coerce")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
    return df


def fetch_kev_cve_ids() -> set[str]:
    """Return the set of CVE IDs with verified, active exploitation."""
    return set(fetch_kev_catalog()["cve_id"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    kev = fetch_kev_catalog(refresh=True)
    print(f"KEV catalog: {len(kev):,} actively exploited CVEs")
    print(f"dateAdded range: {kev.kev_date_added.min().date()} .. "
          f"{kev.kev_date_added.max().date()}")
    print(f"missing dateAdded: {kev.kev_date_added.isna().sum()}")
    print(f"ransomware-linked: {kev.kev_ransomware.sum():,}")
    print()
    print(kev.head())
