import io
import logging
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"


def download_data(destination: str | Path = "data/raw/ai4i2020.csv", force: bool = False) -> Path:
    path = Path(destination)
    if path.exists() and not force:
        logging.info("Using existing dataset %s", path)
        return path
    try:
        request = urllib.request.Request(URL, headers={"User-Agent": "predictmaint-portfolio/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            csv = archive.read("ai4i2020.csv")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(csv)
        logging.info("Downloaded official UCI dataset to %s", path)
        return path
    except Exception as exc:
        raise RuntimeError(
            "UCI download failed. Download from "
            "https://archive.ics.uci.edu/dataset/601/ai4i "
            f"and save ai4i2020.csv at {path}. No substitute data used."
        ) from exc
