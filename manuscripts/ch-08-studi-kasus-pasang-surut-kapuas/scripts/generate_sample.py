#!/usr/bin/env python3
"""
Hasilkan sample CSV pasang surut Cilacap (1 tahun hourly, sintetik realistis).

Karakter yang dipakai:
  - Komponen utama M2 (12,42 jam) amplitudo ~0,40 m (semi-diurnal kuat di Cilacap)
  - Komponen S2 (12,00 jam) amplitudo ~0,10 m
  - Komponen K1 (23,93 jam) amplitudo ~0,20 m (diurnal cukup kuat)
  - Komponen O1 (25,82 jam) amplitudo ~0,10 m
  - Variasi musiman amplitudo (modulasi ~6 bulan) ~10%
  - Noise pengukuran ~0,02 m
  - Gap kecil (~2%) yang harus di-handle pembaca (Bab 6 §6.4)

Tujuan: memberi notebook data out-of-the-box yang pola amplitudonya mirip Cilacap
(GLOSS #291, Indonesia, ~7,75° LS 109,02° BT) tanpa klaim bahwa ini observasi nyata.

Penggunaan:
  python generate_sample.py            # tulis ke ../data/sample/cili_1y_hourly.csv
  python generate_sample.py --days 90  # panjang lain
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HARI_DEFAULT = 365
SEED = 42


def pasang_sintetik(n: int, t0: datetime) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    jam = np.arange(n, dtype=float)
    M2 = 0.40 * np.sin(2 * np.pi * jam / 12.42 + 0.7)
    S2 = 0.10 * np.sin(2 * np.pi * jam / 12.00 + 1.1)
    K1 = 0.20 * np.sin(2 * np.pi * jam / 23.93 + 2.3)
    O1 = 0.10 * np.sin(2 * np.pi * jam / 25.82 + 0.5)
    musiman = 0.08 * np.sin(2 * np.pi * jam / (365.25 * 24))
    noise = 0.02 * rng.standard_normal(n)
    tinggi = (M2 + S2 + K1 + O1 + musiman + noise).round(3)
    waktu = pd.date_range(start=t0, periods=n, freq="h", tz="UTC")
    df = pd.DataFrame({"time": waktu, "tinggi": tinggi})

    # sisipkan ~2% nilai hilang (acak, blok pendek)
    n_gap_blocks = 8
    for _ in range(n_gap_blocks):
        start = int(rng.integers(24, n - 48))
        length = int(rng.integers(1, 6))
        df.loc[start : start + length - 1, "tinggi"] = np.nan
    return df


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", type=int, default=HARI_DEFAULT, help=f"Panjang data dalam hari (default {HARI_DEFAULT}).")
    p.add_argument("--start", default="2024-01-01T00:00:00+00:00", help="Waktu mulai ISO.")
    p.add_argument("--output", type=Path, default=None,
                   help="Path output CSV (default: ../data/sample/cili_1y_hourly.csv).")
    args = p.parse_args()

    t0 = datetime.fromisoformat(args.start)
    df = pasang_sintetik(args.days * 24, t0)
    out = args.output or (Path(__file__).resolve().parent.parent / "data" / "sample" / "cili_1y_hourly.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Tulis {len(df)} baris ke {out} (NaN: {int(df['tinggi'].isna().sum())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
