#!/usr/bin/env python3
"""
Unduh data pasang surut dari sumber terbuka untuk studi kasus Bab 8.

Sumber yang didukung:
  - ioc     : UNESCO/IOC Sea Level Station Monitoring Facility (real-time, max 30 hari/request)
  - uhslc   : UHSLC ERDDAP OPeDAP (research quality hourly/daily)
  - psmsl   : PSMSL (rata-rata MSL bulanan jangka panjang)

Penggunaan dasar:
  python download_ioc.py --source ioc --code cili --days 30 --output out.csv
  python download_ioc.py --source ioc --code cili --start 2024-01-01 --end 2024-12-31 --output out.csv
  python download_ioc.py --source uhslc --station_id <ID> --start 2023-01-01 --end 2024-12-31 --output out.csv
  python download_ioc.py --source psmsl --station_id 2199 --output out.csv

Lihat README.md di folder ini untuk daftar station, lisensi, dan catatan atribusi.
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

IOC_BASE = "https://www.ioc-sealevelmonitoring.org"
UHSLC_ERDDAP = "https://uhslc.soest.hawaii.edu/erddap/tabledap"
PSMSL_BASE = "https://psmsl.org/data/obtaining"

USER_AGENT = "Buku-DL-Meteorologi/1.0 (research; +https://kanadakurniawan.com)"
TIMEOUT = 60


def _log(msg: str) -> None:
    print(f"[{datetime.utcnow().isoformat(timespec='seconds')}Z] {msg}", file=sys.stderr)


def fetch_ioc_block(code: str, end: datetime, days: int) -> pd.DataFrame:
    """Ambil satu blok (<=30 hari) dari IOC bgraph.php endpoint."""
    url = f"{IOC_BASE}/bgraph.php"
    params = {"code": code, "output": "tab", "period": str(days)}
    _log(f"GET IOC code={code} days={days} ending~{end.date()}")
    r = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    r.raise_for_status()
    lines = [ln for ln in r.text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise RuntimeError(f"Tidak ada data dari IOC untuk code={code} (cek station aktif).")
    header = lines[0].split("\t")
    rows = [ln.split("\t") for ln in lines[1:]]
    df = pd.DataFrame(rows, columns=header)
    time_col = next((c for c in df.columns if c.lower().startswith("time") or "utc" in c.lower()), None)
    if time_col is None:
        raise RuntimeError(f"Kolom waktu tidak ditemukan. Header: {list(df.columns)}")
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df = df.dropna(subset=[time_col]).rename(columns={time_col: "time"})
    sensor_cols = [c for c in df.columns if c != "time"]
    for c in sensor_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["tinggi"] = df[sensor_cols].mean(axis=1, skipna=True)
    return df[["time", "tinggi"] + sensor_cols].sort_values("time").reset_index(drop=True)


def download_ioc(code: str, output: Path, days: int = 30,
                 start: str | None = None, end: str | None = None) -> None:
    """Unduh IOC real-time. Jika start/end diberikan, lakukan loop blok 30 hari."""
    if start or end:
        start_dt = datetime.fromisoformat(start) if start else datetime.utcnow() - timedelta(days=365)
        end_dt = datetime.fromisoformat(end) if end else datetime.utcnow()
        cur_end = end_dt
        frames = []
        while cur_end > start_dt:
            block_end = cur_end
            block_start = max(block_end - timedelta(days=days), start_dt)
            actual_days = max(1, (block_end - block_start).days)
            df = fetch_ioc_block(code, end=block_end, days=actual_days)
            df = df[df["time"] >= pd.Timestamp(block_start, tz="UTC")]
            frames.append(df)
            cur_end = block_start - timedelta(seconds=1)
            time.sleep(1.0)
        out = pd.concat(frames, ignore_index=True).drop_duplicates(subset="time").sort_values("time")
    else:
        out = fetch_ioc_block(code, end=datetime.utcnow(), days=days)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    _log(f"Simpan {len(out)} baris ke {output}")


def download_uhslc(station_id: str, output: Path, start: str, end: str,
                   dataset: str = "global_hourly_rqds") -> None:
    """Unduh via ERDDAP OPeDAP CSV."""
    dataset_path = f"{UHSLC_ERDDAP}/{dataset}.csv"
    params = {
        "station_id": f'"{station_id}"',
        "time>=": start,
        "time<=": end,
    }
    _log(f"GET UHSLC dataset={dataset} station_id={station_id} {start}..{end}")
    r = requests.get(dataset_path, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    r.raise_for_status()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(r.text)
    n_lines = sum(1 for _ in r.text.splitlines()) - 1
    _log(f"Simpan {n_lines} baris ke {output}")


def download_psmsl(station_id: str, output: Path) -> None:
    """Unduh data PSMSL untuk satu station (format metric atau RLR)."""
    url = f"{PSMSL_BASE}/stations/{station_id}.php"
    _log(f"GET PSMSL station_id={station_id} (halaman metadata; unduh manual lewat UI)")
    _log(f"  → {url}")
    _log("  Skrip ini hanya menulis URL & membuka halaman; data metric/RLR")
    _log("  perlu diunduh manual dari halaman station (tautan 'Metric' / 'RLR').")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(f"# Buka halaman berikut untuk unduh manual:\n# {url}\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["ioc", "uhslc", "psmsl"], required=True)
    p.add_argument("--code", help="Kode IOC (mis. cili, ambon, bitu).")
    p.add_argument("--station_id", help="Station ID UHSLC atau PSMSL.")
    p.add_argument("--days", type=int, default=30, help="Periode hari untuk IOC (default 30).")
    p.add_argument("--start", help="Tanggal mulai (ISO YYYY-MM-DD).")
    p.add_argument("--end", help="Tanggal akhir (ISO YYYY-MM-DD).")
    p.add_argument("--dataset", default="global_hourly_rqds",
                   help="Dataset UHSLC ERDDAP (default: global_hourly_rqds).")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    try:
        if args.source == "ioc":
            if not args.code:
                raise SystemExit("--code wajib untuk --source ioc")
            download_ioc(args.code, args.output, days=args.days, start=args.start, end=args.end)
        elif args.source == "uhslc":
            if not (args.station_id and args.start and args.end):
                raise SystemExit("--station_id, --start, --end wajib untuk --source uhslc")
            download_uhslc(args.station_id, args.output, args.start, args.end, dataset=args.dataset)
        elif args.source == "psmsl":
            if not args.station_id:
                raise SystemExit("--station_id wajib untuk --source psmsl")
            download_psmsl(args.station_id, args.output)
        return 0
    except requests.HTTPError as e:
        _log(f"GAGAL HTTP {e.response.status_code}: {e.response.text[:200]}")
        return 1
    except Exception as e:
        _log(f"GAGAL: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
