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
| Tabel | Tabel 1.3 | Glosarium mini bab 1 | — | §1.12 |
| Persamaan | — (belum ada) | | | |
| Kode | Kode 1.1 | Verifikasi TensorFlow + GPU | — | §1.6 |
| Kode | Kode 1.2 | Pembuatan tensor suhu_hari | — | §1.7 |
| Kode | Kode 1.3 | Mini-challenge persistence vs klimatologis | — | §1.7b |

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

| Jenis | Nomor | Caption / isi | File | Dirujuk di § |
|---|---|---|---|---|
| Gambar | Gambar 4.1 | Contoh learning curve (train turun, val naik → overfit) | `figures/fig-4-1-learning-curve.png` | §4.7 |
| Tabel | Tabel 4.1 | Perbandingan fungsi aktivasi dari sisi gradien | — | §4.3 |
| Tabel | Tabel 4.2 | SGD vs Adam | — | §4.4 |
| Persamaan | (4.1) | $w \leftarrow w - \eta \frac{\partial L}{\partial w}$ | — | §4.1 |
| Persamaan | (4.2) | aturan rantai backprop | — | §4.2 |
| Persamaan | (4.3) | $\sigma'(z)=\sigma(z)(1-\sigma(z))$ | — | §4.3 |
| Kode | Kode 4.1 | GradientTape manual backprop | — | §4.2 |
| Kode | Kode 4.2 | LR & batch di Keras | — | §4.5 |
| Kode | Kode 4.3 | Callback (EarlyStopping, Checkpoint, ReduceLROnPlateau + scheduler) | — | §4.6 |

### Bab 5 — Overfitting, Regularisasi dan Evaluasi untuk Data Iklim

| Jenis | Nomor | Caption / isi | File | Dirujuk di § |
|---|---|---|---|---|
| Gambar | Gambar 5.1 | Learning curve overfit | `figures/fig-5-1-learning-curve.png` | §5.2 |
| Tabel | Tabel 5.1 | Underfit / fit / overfit | — | §5.1 |
| Tabel | Tabel 5.2 | Panduan memilih metrik regresi | — | §5.4 |
| Tabel | Tabel 5.3 | Kuartet verifikasi WMO (POD/FAR/CSI/TS) | — | §5.4 |
| Tabel | Tabel 5.4 | Dua model, cerita metrik berbeda | — | §5.4 |
| Tabel | Tabel 5.5 | Skema walk-forward (5 fold) | — | §5.5 |
| Persamaan | (5.1) | L2: $\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \sum w_j^2$ | — | §5.3 |
| Kode | Kode 5.1 | EarlyStopping | — | §5.3 |
| Kode | Kode 5.2 | Arsitektur + L2 + dropout | — | §5.3 |
| Kode | Kode 5.3 | Walk-forward sederhana | — | §5.5 |

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
| [8] | `wheeler2004rmm` | Artikel | 10.1175/1520-0493(2004)132<1917:AARMMI>2.0.CO;2 | ✅ |
| [9] | `lestari2019jakarta` | Artikel | 10.1016/j.wace.2019.100202 | ✅ |
| [10] | `jolliffe2011forecast` | Buku | 10.1002/9781119960003 | ✅ |

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

### Bab 4 — Backpropagation, Optimasi dan Pelatihan

| `[n]` | Key `refs.bib` | Jenis | DOI / ISBN / arXiv | Status |
|---|---|---|---|---|
| [1] | `rumelhart1986learning` | Artikel | 10.1038/323533a0 | ✅ |
| [2] | `kingma2015adam` | Artikel (ICLR) | arXiv:1412.6980 | ✅ |
| [3] | `goodfellow2016deep` | Buku | (MIT Press) | ✅ |
| [4] | `abadi2016tensorflow` | Software/arXiv | arXiv:1603.04467 | ✅ |

### Bab 5 — Overfitting, Regularisasi dan Evaluasi

| `[n]` | Key `refs.bib` | Jenis | DOI / ISBN / arXiv | Status |
|---|---|---|---|---|
| [1] | `wmo2018verification` | Tech report (WMO) | (pdf WMO) | ✅ |
| [2] | `goodfellow2016deep` | Buku | (MIT Press) | ✅ |
| [3] | `srivastava2014dropout` | Artikel (JMLR) | (JMLR) | ✅ |
| [4] | `gupta2009decomposition` | Artikel (J. Hydrol) | 10.1016/j.jhydrol.2009.08.003 | ✅ |

---

## Catatan Pemeliharaan

- **Saat menambah aset/sitasi baru di `master.md`:** perbarui file ini segera, sebelum
  menutup bab tersebut.
- **Nomor aset & sitasi mengikuti `master.md`** (sumber kebenaran). Jangan menomori ulang
  hanya di register.
- `CHECKLIST-EVALUASI.md` Fase 2 bagian C (sitasi) & E (build) menggunakan register ini
  sebagai input ke cek otomatis.