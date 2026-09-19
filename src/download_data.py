"""데이터 다운로드"""

from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "Life Expectancy Data.csv"
DATA_URL = "https://www.kaggle.com/api/v1/datasets/download/kumarajarshi/life-expectancy-who"


def download_data():
    if DATA_PATH.exists():
        print(f"Already present: {DATA_PATH.name}")
        return DATA_PATH
    request = Request(DATA_URL, headers={"User-Agent": "life-expectancy-ann/1.0"})
    try:
        with urlopen(request, timeout=60) as response:
            content = response.read()
        with ZipFile(BytesIO(content)) as archive:
            csv = archive.read("Life Expectancy Data.csv")
    except (OSError, BadZipFile, KeyError) as exc:
        raise RuntimeError(
            "Download failed. Download Life Expectancy Data.csv from "
            "https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who "
            "and put it in data/."
        ) from exc
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_bytes(csv)
    print(f"Downloaded: {DATA_PATH.name}")
    return DATA_PATH


if __name__ == "__main__":
    download_data()
