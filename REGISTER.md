# Register Aset & Sitasi — Buku *Pengantar Deep Learning untuk Meteorologi*

> **Inventaris terpusat** untuk penomoran aset (gambar, tabel, persamaan, kode) dan
> sitasi per bab. Dipakai saat evaluasi internal (Fase 2 — lihat `CHECKLIST-EVALUASI.md`).
>
> **Aturan penomoran (konvensi akademik):** tiap jenis aset memiliki deret nomor
> **sendiri**, reset di tiap bab. Format: bab.jenis-urutan → `Gambar 2.1`, `Tabel 2.1`,
> `Persamaan (2.1)`, `Kode 2.1`.
>
> **Aturan rujukan:** setiap aset **harus dirujuk** di teks (§ terkait); nomor aset tidak
> boleh ganda; nomor tidak boleh loncat (0,1,2).

---

## Bagian A — Register Penomoran Aset

### Bab 1 — Pengantar: Deep Learning untuk Meteorologi

| Jenis | Nomor | Caption / isi | File | Dirujuk di § |
|---|---|---|---|---|
| Gambar | Gambar 1.1 | Keterkaitan AI, ML, dan DL | `figures/fig-1-1-hierarki-ai-by-chatgpt.png` | §1.1 |
| Tabel | Tabel 1.1 | Peta aplikasi DL dalam meteorologi | — | §1.3 |
| Tabel | Tabel 1.2 | Contoh data cuaca mini → shape tensor | — | §1.7 |
| Persamaan | — (belum ada) | | | |
| Kode | — (belum bernomor) | Verifikasi TensorFlow/GPU; tensor `suhu_hari` | — | §1.6–1.7 |

> **Catatan:** Kode Verifikasi & tensor di §1.6/1.7 belum diberi nomor (`Kode 1.1`, dst).
> Perlu ditambahkan saat review.

### Bab 2 — Regresi: Perceptron dan Jaringan Saraf

| Jenis | Nomor | Caption / isi | File | Dirujuk di § |
|---|---|---|---|---|
| Gambar | Gambar 2.1 | Struktur neuron buatan (x → z → aktivasi a) | `figures/fig-2-1-neuron.png` | §2.2 |
| Tabel | Tabel 2.1 | Contoh target regresi meteorologi (satuan & sifat data) | — | §2.1 |
| Tabel | Tabel 2.2 | Contoh windowing (dua langkah) pasang surut | — | §2.5 |
| Tabel | Tabel 2.3 | Perbandingan MAE vs MSE | — | §2.6 |
| Persamaan | (2.1) | $z = \sum w_i x_i + b$ | — | §2.2 |
| Persamaan | (2.2) | $a = f(z)$ | — | §2.2 |
| Persamaan | (2.3) | $\hat{y} = wx + b$ | — | §2.3 |
| Persamaan | (2.4) | $\mathrm{ReLU}(x) = \max(0, x)$ | — | §2.4 |
| Persamaan | (2.5) | $\mathrm{MAE} = \frac{1}{n}\sum \|\cdot\|$ (contoh) | — | §2.6 |
| Persamaan | (2.6) | $\mathrm{MSE} = \frac{1}{n}\sum (\cdot)^2$ (contoh) | — | §2.6 |
| Kode | Kode 2.1 | Definisi arsitektur MLP regresi (Keras) | — | §2.4 |
| Kode | Kode 2.2 | Windowing + split berbasis waktu + data sintetik | — | §2.5 |
| Kode | Kode 2.3 | Compile, latih, evaluasi vs persistence | — | §2.5 |

> **Catatan:** Kode/notebook pendamping (setup data, baseline, training) di Bab 2 belum
> diberi nomor `Kode 2.2` dst. Tambahkan saat review.

### Bab 3 — Klasifikasi: Mengenali Kategori Fenomena Cuaca

| Jenis | Nomor | Caption / isi | File | Dirujuk di § |
|---|---|---|---|---|
| Gambar | Gambar 3.1 | Kurva sigmoid memetakan z ke (0,1) | `figures/fig-3-1-sigmoid.png` | §3.2 |
| Gambar | Gambar 3.2 | Confusion matrix contoh data tidak seimbang | `figures/fig-3-2-confusion-matrix.png` | §3.6 |
| Tabel | Tabel 3.1 | Perbedaan regresi vs klasifikasi | — | §3.1 |
| Tabel | Tabel 3.2 | Perbandingan sigmoid vs softmax | — | §3.3 |
| Tabel | Tabel 3.3 | Contoh data tidak seimbang | — | §3.6 |
| Tabel | Tabel 3.4 | Struktur confusion matrix biner | — | §3.8, §3.6 |
| Persamaan | (3.1) | $\sigma(z) = \frac{1}{1+e^{-z}}$ | — | §3.2 |
| Persamaan | (3.2) | softmax | — | §3.3 |
| Persamaan | (3.3) | binary cross-entropy | — | §3.4 |
| Persamaan | (3.4) | F1 | — | §3.6 |
| Kode | Kode 3.1 | Model klasifikasi biner hujan/tidak (Keras) | — | §3.5 |
| Kode | Kode 3.2 | Model klasifikasi multi-kelas intensitas | — | §3.5 |
| Kode | Kode 3.3 | Fit dengan class_weight untuk imbalance | — | §3.5 |

### Bab 4 — Backpropagation, Optimasi dan Pelatihan

*(belum ditulis)*

---

## Bagian B — Register Sitasi

> Kolom **status**: ✅ terverifikasi (DOI/ISBN/arXiv cek) · ⚠️ perlu cek-banding · ✍️ draft
> Aturan (dari `outline.md` → Kriteria Sitasi): `[n]` di teks ≡ `References` ≡ `refs.bib`;
> urutan incremental pertama-muncul; DOI valid; ISBN/arXiv bila tanpa DOI.

### Bab 1 — Pengantar

| `[n]` | Key `refs.bib` | Jenis | DOI / ISBN / arXiv | Status |
|---|---|---|---|---|
| [1] | `russell2021ai` | Buku | ISBN 978-0134610993 | ✅ |
| [2] | `mitchell1997ml` | Buku | ISBN 978-0070428072 | ✅ |
| [3] | `lecun2015deep` | Artikel | 10.1038/nature14539 | ✅ |
| [4] | `goodfellow2016deep` | Buku | (MIT Press) | ✅ |
| [5] | `rosenblatt1958perceptron` | Artikel | 10.1037/h0042519 | ✅ |
| [6] | `abadi2016tensorflow` | Software/arXiv | arXiv:1603.04467 | ✅ |
| [7] | `reichstein2019deep` | Artikel | 10.1038/s41586-019-0912-1 | ✅ |

### Bab 2 — Regresi

| `[n]` | Key `refs.bib` | Jenis | DOI / ISBN / arXiv | Status |
|---|---|---|---|---|
| [1] | `rosenblatt1958perceptron` | Artikel | 10.1037/h0042519 | ✅ |
| [2] | `krizhevsky2012imagenet` | Artikel (NeurIPS) | 10.1145/3065386 | ✅ |
| [3] | `big_pasut` | Dataset/web | URL tides.big.go.id, diakses Sep 2026 | ✅ |

### Bab 3 — Klasifikasi

| `[n]` | Key `refs.bib` | Jenis | DOI / ISBN / arXiv | Status |
|---|---|---|---|---|
| [1] | `goodfellow2016deep` | Buku | (MIT Press) | ✅ |
| [2] | `rosenblatt1958perceptron` | Artikel | 10.1037/h0042519 | ✅ |
| [3] | `abadi2016tensorflow` | Software/arXiv | arXiv:1603.04467 | ✅ |
| [4] | `wmo2018verification` | Tech report (WMO) | (pdf WMO) | ✅ |

---

## Catatan Pemeliharaan

- **Saat menambah aset/sitasi baru di `master.md`:** perbarui file ini segera, sebelum
  menutup bab tersebut.
- **Nomor aset & sitasi mengikuti `master.md`** (sumber kebenaran). Jangan menomori ulang
  hanya di register.
- `CHECKLIST-EVALUASI.md` Fase 2 bagian C (sitasi) & E (build) menggunakan register ini
  sebagai input ke cek otomatis.