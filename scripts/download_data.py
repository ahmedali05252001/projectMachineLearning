"""Download public football-data.co.uk CSVs used by the notebook."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.soccer_ml import DEFAULT_DIVS, DEFAULT_SEASONS, download_csv, season_csv_url


def main() -> None:
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for season in DEFAULT_SEASONS:
        for div in DEFAULT_DIVS:
            url = season_csv_url(season, div)
            dest = raw_dir / f"{season}_{div}.csv"
            print(f"Downloading {url} -> {dest}")
            download_csv(url, dest)

    print("Done.")


if __name__ == "__main__":
    main()
