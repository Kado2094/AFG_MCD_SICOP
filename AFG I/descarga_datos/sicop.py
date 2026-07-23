import os
import time
from datetime import date

import requests
from dateutil.relativedelta import relativedelta

BASE = "https://dlsaobservatorioprod.blob.core.windows.net/fs-synapse-observatorio-produccion/Zip/{yyyymm}.zip"

def month_range(start: date, end: date):
    """Yield first day of each month from start to end inclusive."""
    cur = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while cur <= last:
        yield cur
        cur = (cur + relativedelta(months=1))

def download_file(url: str, out_path: str, timeout=60, retries=3):
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if r.status_code == 404:
                    return "missing"  # no existe ese mes (o aún no publicado)
                r.raise_for_status()
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                tmp = out_path + ".part"
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp, out_path)
                return "ok"
        except Exception as e:
            if attempt == retries:
                return f"error: {e}"
            time.sleep(2 * attempt)

def main():
    # Ajuste su rango aquí
    start = date(2010, 1, 1)
    end   = date(2025, 11, 1)

    out_dir = "./sicop_zips"
    pause_seconds = 1.0

    for d in month_range(start, end):
        yyyymm = f"{d.year}{d.month:02d}"
        url = BASE.format(yyyymm=yyyymm)
        out_path = os.path.join(out_dir, f"{yyyymm}.zip")

        if os.path.exists(out_path):
            print(f"[SKIP] {yyyymm} ya existe")
            continue

        status = download_file(url, out_path)
        print(f"[{status.upper()}] {yyyymm} -> {url}")

        time.sleep(pause_seconds)

if __name__ == "__main__":
    main()
