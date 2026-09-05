# Skrip unduh data pasang surut — Bab 8

> **Bookend**: `Pengantar Deep Learning untuk Meteorologi` — Bab 8 (Pasang Surut).

Skrip Python sederhana untuk mengunduh data pasang surut dari tiga sumber terbuka
untuk studi kasus Cilacap (atau station Indonesia lain). Skrip ini **tidak** menyimpan
data ke dalam repo (lihat `.gitignore` untuk pengecualian); ia hanya menulis ke
`data/raw/` lokal Anda.

## Sumber

| Sumber | Fungsi | Lisensi | Atribusi |
|---|---|---|---|
| UNESCO/IOC Sea Level Station Monitoring Facility | Real-time / near real-time (max 30 hari/request) | Gratis riset/pendidikan | "IOC Sea Level Station Monitoring Facility" |
| UHSLC (University of Hawaii Sea Level Center) | Hourly & daily research-quality (arsip panjang) | Gratis riset/pendidikan | "University of Hawaii Sea Level Center" |
| PSMSL | Rata-rata MSL bulanan jangka panjang | Gratis riset dengan sitasi Holgate (2013) | Holgate, S. J. (2013), J. Coast. Res. 29(3) |

## Instalasi

Hanya butuh Python 3.9+ dengan `requests` dan `pandas`:

```bash
pip install requests pandas
```

Untuk UHSLC ERDDAP, library `xarray`/`netCDF4` berguna tapi tidak wajib (skrip export CSV).

## Cara pakai

### 1. IOC (real-time, max 30 hari)

```bash
# Cilacap, 30 hari terakhir (default)
python scripts/download_ioc.py --source ioc --code cili --days 30 \
    --output data/raw/cili_30d.csv

# Station lain (mis. Ambon, Bitung)
python scripts/download_ioc.py --source ioc --code ambon --days 30 \
    --output data/raw/ambon_30d.csv
```

Untuk periode lebih panjang dari 30 hari, gunakan `--loop` dengan tanggal mulai/akhir
(skrip otomatis memecah menjadi blok 30 hari dan menggabungkan):

```bash
python scripts/download_ioc.py --source ioc --code cili \
    --start 2024-01-01 --end 2024-12-31 \
    --output data/raw/cili_2024_hourly.csv
```

### 2. UHSLC (hourly research quality, periode panjang)

```bash
python scripts/download_ioc.py --source uhslc --station_id <ID> \
    --start 2023-01-01 --end 2024-12-31 \
    --output data/raw/cili_uhslc_2023_2024.csv
```

Kode station UHSLC bisa dilihat di metadata IOC atau di
`https://uhslc.soest.hawaii.edu/data/`. Untuk Indonesia yang biasanya muncul:
Ambon, Bitung, dan beberapa station lain yang masuk dataset JASL/UHSLC.

### 3. PSMSL (MSL bulanan jangka panjang)

```bash
python scripts/download_ioc.py --source psmsl --station_id 2199 \
    --output data/raw/cilacap_psmsl_monthly.csv
```

PSMSL ID untuk station Indonesia umum: 1709 (Bitung II), 1752 (Sibolga II),
2193 (Padang B), 2195 (Sabang), 2197 (Prigi), 2199 (Cilacap B), 2200 (Benoa B),
2274 (Saumlaki). Daftar lengkap: <https://psmsl.org/data/obtaining/>.

## Catatan

- **Toleransi rate-limit**: skrip menunggu 1 detik antar-request IOC; untuk periode
  panjang, ini bisa memakan waktu beberapa menit. Tidak ada rate-limit keras di
  endpoint publik, tapi bersikap sopan adalah praktik baik.
- **Sampling IOC**: data dapat 1-menit, 3-menit, atau hourly tergantung station.
  Skrip menyimpan semua kolom sensor yang diterima; buku hanya menggunakan
  kolom utama (rata-rata antar-sensor).
- **Sampling UHSLC**: hourly atau 1-menit tergantung dataset; pilih via argumen
  `--dataset` (default: `global_hourly_rqds`).
- **PSMSL**: hanya data bulanan MSL; tidak cocok untuk prakiraan hourly.

## Verifikasi cepat

Setelah unduh, periksa file:

```python
import pandas as pd
df = pd.read_csv("data/raw/cili_30d.csv", parse_dates=["time"])
print(df.head(), df.shape, df.isna().sum())
```

## Skrip lain di folder ini

| Skrip | Fungsi |
|---|---|
| `download_ioc.py` | Unduh data IOC real-time / UHSLC ERDDAP / metadata PSMSL (skrip utama) |
| `generate_sample.py` | Hasilkan sample CSV Cilacap sintetik deterministik |
| `generate_figures.py` | Hasilkan Gambar 8.2 (prediksi vs aktual) & 8.3 (residu) dari sample |

## Lisensi skrip

Kode skrip mengikuti lisensi buku (lihat `LICENSE` di root repo). Data yang
diunduh tunduk pada lisensi masing-masing sumber; cantumkan atribusi saat
menyebarluaskan hasil.
