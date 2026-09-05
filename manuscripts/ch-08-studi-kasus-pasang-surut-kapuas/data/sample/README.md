# Sample Data Pasang Surut — Bab 8

> **PENTING**: Berkas di folder ini adalah **sampel sintetik** yang merepresentasikan
> karakter pasang surut station Cilacap (campuran condong semi-diurnal) untuk
> demonstrasi notebook out-of-the-box. Untuk eksperimen nyata, **ganti dengan data
> hasil unduh** dari `scripts/download_ioc.py` (IOC) atau `download_uhslc` (UHSLC).

## Daftar Berkas

| Berkas | Ukuran | Sumber | Lisensi | Catatan |
|---|---|---|---|---|
| `cili_1y_hourly.csv` | ~1 tahun hourly (~8.760 baris) | Sintetik (lihat header) | CC-BY-4.0 buku | Realistis tapi bukan observasi |

## Format

CSV dengan kolom:

- `time` — datetime UTC, hourly
- `tinggi` — tinggi muka air dalam meter (relatif terhadap mean)
- `sensor_1`, `sensor_2`, ... — kolom sensor tiruan (untuk konsistensi format
  dengan output IOC/UHSLC); kosong untuk sample sintetik

## Mengapa sintetik?

- **Reproducibility**: pembaca dapat menjalankan notebook tanpa akses internet
  atau API key apa pun.
- **Tidak ada data asli yang hilang**: bila skrip unduh gagal (mis. station IOC
  sedang non-aktif), notebook tetap bisa dijalankan untuk verifikasi alur.
- **Hasil angka tidak boleh dikutip** sebagai performa model di station nyata
  (lihat Ringkasan §8.6 di `master.md` — keterbatasan eksplisit).

## Cara mengganti dengan data nyata

1. Install dependensi: `pip install requests pandas`
2. Unduh data Cilacap 30 hari dari IOC:
   ```
   python ../scripts/download_ioc.py --source ioc --code cili --days 30 \
       --output ../data/raw/cili_30d.csv
   ```
3. Ubah path di `master.md` Kode 8.1 (atau notebook `ch-08-07_studi_kasus_pasang_surut.ipynb`)
   dari `data/sample/cili_1y_hourly.csv` ke `data/raw/cili_30d.csv`.
4. Jalankan notebook ulang. Bandingkan hasil dengan baseline untuk validasi.

## Pembuatan sample

Sample dihasilkan oleh `scripts/generate_sample.py` (lihat folder `scripts/`).
Proses deterministik (seed tetap) sehingga setiap clone repo menghasilkan CSV yang sama.
