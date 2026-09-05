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

### Bab 6 — Data Meteorologi: Sumber, Kualitas dan Persiapan

| Jenis | Nomor | Caption / isi | File | Dirujuk di § |
|---|---|---|---|---|
| Gambar | Gambar 6.1 | Distribusi curah hujan harian (ekor panjang) | `figures/fig-6-1-distribusi-hujan.png` | §6.5 |
| Tabel | Tabel 6.1 | Sumber data utama (BMKG, ERA5, CMIP6, PSMSL, satelit) | — | §6.2 |
| Tabel | Tabel 6.2 | Perbandingan format berkas (CSV/NetCDF/GRIB) | — | §6.3 |
| Persamaan | (6.1) | z-score normalisasi (train) | — | §6.7 |
| Persamaan | (6.2) | transformasi target log1p | — | §6.7 |
| Kode | Kode 6.1 | Membaca NetCDF dengan xarray | — | §6.3 |
| Kode | Kode 6.2 | Membaca GRIB dengan cfgrib | — | §6.3 |
| Kode | Kode 6.3 | Menyatukan ERA5 per jam → tabel harian | — | §6.3 |
| Kode | Kode 6.4 | Cek & isi nilai hilang (pandas) | — | §6.4 |
| Kode | Kode 6.5 | Dekomposisi musiman & korelasi silang | — | §6.5 |
| Kode | Kode 6.6 | Fitur lag + musiman + ENSO/MJO | — | §6.6 |
| Kode | Kode 6.7 | Normalisasi (skala train) + split waktu | — | §6.7 |
| Kode | Kode 6.8 | Menyimpan X/y utk bab berikutnya | — | §6.7 |

### Bab 7 — Deret Waktu dan Model Sekuensial: RNN, LSTM, GRU

| Jenis | Nomor | Caption / isi | File | Dirujuk di § |
|---|---|---|---|---|
| Gambar | Gambar 7.1 | Ilustrasi RNN unrolled (state h) | `figures/fig-7-1-rnn-unrolled.png` | §7.4 |
| Tabel | Tabel 7.1 | Contoh windowing (w=3, h=1) | — | §7.2 |
| Tabel | Tabel 7.2 | Pilihan panjang window | — | §7.2 |
| Tabel | Tabel 7.3 | Baseline deret waktu | — | §7.3 |
| Tabel | Tabel 7.4 | LSTM vs GRU | — | §7.6 |
| Tabel | Tabel 7.5 | Strategi multi-langkah | — | §7.7 |
| Persamaan | (7.1) | input shape (batch, waktu, fitur) | — | §7.2 |
| Persamaan | (7.2) | $h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$ | — | §7.4 |
| Persamaan | (7.3) | LSTM gate lupa $f_t$ | — | §7.5 |
| Persamaan | (7.4) | LSTM gate masukan $i_t$ | — | §7.5 |
| Persamaan | (7.5) | LSTM gate keluaran $o_t$ | — | §7.5 |
| Persamaan | (7.6) | skill score SS | — | §7.8 |
| Kode | Kode 7.1 | buat_window (window→X,y) | — | §7.2 |
| Kode | Kode 7.2 | baseline persistence & klimatologi | — | §7.3 |
| Kode | Kode 7.3 | LSTM univariate | — | §7.7 |
| Kode | Kode 7.4 | LSTM multivariate | — | §7.7 |
| Kode | Kode 7.5 | Plot forecast vs aktual | — | §7.8 |

### Bab 8 — Studi Kasus: Pasang Surut Indonesia (Contoh Cilacap)

| Jenis | Nomor | Caption / isi | File | Dirujuk di § |
|---|---|---|---|---|
| Gambar | Gambar 8.1 | Spektrum frekuensi pasang surut (M2/K1) | `figures/fig-8-1-spektrum-pasang.png` | §8.2 |
| Gambar | Gambar 8.2 | Prediksi vs aktual 7 hari (data sample Cilacap) | `figures/fig-8-2-forecast-7hari.png` | §8.5 |
| Gambar | Gambar 8.3 | Residu per amplitudo dan fase pasang M2 | `figures/fig-8-3-residu.png` | §8.5 |
| Tabel | Tabel 8.1 | Tipe pasang surut Indonesia | — | §8.2 |
| Tabel | Tabel 8.2 | Harmonik vs machine learning | — | §8.2 |
| Tabel | Tabel 8.3 | Station Indonesia di sumber terbuka (IOC/UHSLC/PSMSL) | — | §8.3 |
| Tabel | Tabel 8.3a | Ringkasan dataset Cilacap yang dibangun | — | §8.3 |
| Tabel | Tabel 8.4 | Pilihan window (jam-an) | — | §8.4 |
| Tabel | Tabel 8.5 | Contoh hasil MAE per horizon | — | §8.5 |
| Tabel | Tabel 8.6 | Skill score relatif vs persistence | — | §8.5 |
| Persamaan | — (belum ada nomor eksplisit) | — | — | — |
| Kode | Kode 8.1 | Setup & pemuatan data (sample/raw/sintetik) | — | §8.3 |
| Kode | Kode 8.2 | Kerangka model MLP/LSTM/GRU | — | §8.4 |
| Kode | Kode 8.3 | Simulasi pengisian gap | — | §8.6 |
| Data | (sample) | Sample CSV Cilacap 1 tahun hourly sintetik | `data/sample/cili_1y_hourly.csv` | §8.3, Kode 8.1 |
| Skrip | `scripts/download_ioc.py` | Unduh IOC/UHSLC/PSMSL | — | §8.3 |
| Skrip | `scripts/generate_sample.py` | Hasilkan sample CSV | — | §8.3 |
| Skrip | `scripts/generate_figures.py` | Hasilkan Gambar 8.2 & 8.3 dari sample | — | §8.5 |

### Bab 9 — Studi Kasus: Curah Hujan Stasiun BMKG

| Jenis | Nomor | Caption / isi | File | Dirujuk di § |
|---|---|---|---|---|
| Gambar | Gambar 9.1 | Precision-recall untuk hujan lebat | `figures/fig-9-1-precision-recall.png` | §9.4 |
| Gambar | Gambar 9.2 | Verifikasi per kategori intensitas | `figures/fig-9-2-verifikasi-kategori.png` | §9.7 |
| Tabel | Tabel 9.1 | Fitur yang dibangun untuk stasiun | — | §9.2 |
| Tabel | Tabel 9.2 | Kategori intensitas hujan | — | §9.3 |
| Tabel | Tabel 9.3 | Verifikasi threshold (POD/FAR/CSI) | — | §9.4 |
| Tabel | Tabel 9.4 | Rancangan eksperimen | — | §9.5 |
| Tabel | Tabel 9.5 | Verifikasi per kategori | — | §9.7 |
| Persamaan | (9.1) | POD/FAR/CSI | — | §9.4 |
| Kode | Kode 9.1 | Verifikasi CSI/POD/FAR di threshold | — | §9.4 |
| Kode | Kode 9.2 | Permutation importance | — | §9.6 |
| Kode | Kode 9.3 | Crosstab kategori | — | §9.7 |

### Bab 10 — Operasional & Arah Riset

| Jenis | Nomor | Caption / isi | File | Dirujuk di § |
|---|---|---|---|---|
| Gambar | Gambar 10.1 | Grafik kendali MAE (deteksi drift) | `figures/fig-10-1-control-chart.png` | §10.2 |
| Tabel | Tabel 10.2 | Alur keputusan retraining | — | §10.3 |
| Tabel | Tabel 10.3 | Etika pengguna | — | §10.6 |
| Kode | Kode 10.1 | Ensembel multi-seed | — | §10.4 |
| Kode | Kode 10.2 | Regresi kuantil | — | §10.4 |
| Kode | Kode 10.3 | SHAP pada Keras | — | §10.5 |

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

### Bab 6 — Data Meteorologi

| `[n]` | Key `refs.bib` | Jenis | DOI / ISBN / arXiv | Status |
|---|---|---|---|---|
| [1] | `bmkg_dataonline` | Web (BMKG) | URL dataonline.bmkg.go.id | ✅ |
| [2] | `c3s_era5` | Web (C3S) | URL cds.climate.copernicus.eu | ✅ |
| [3] | `hersbach2020era5` | Artikel | 10.1002/qj.3803 | ✅ |
| [4] | `psmsl` | Web (PSMSL) | URL psmsl.org | ✅ |
| [5] | `big_tides` | Web (BIG) | URL tides.big.go.id | ✅ |
| [6] | `funk2015chirps` | Artikel (Sci Data) | 10.1038/sdata.2015.66 | ✅ |
| [7] | `wolter1998mei` | Prosiding | (17th Climate Diagnostics) | ✅ |
| [8] | `wheeler2004rmm` | Artikel | 10.1175/1520-0493(2004)132<1917:AARMMI>2.0.CO;2 | ✅ |
| [9] | `goodfellow2016deep` | Buku | (MIT Press) | ✅ |
| [10] | `jolliffe2011forecast` | Buku | 10.1002/9781119960003 | ✅ |

### Bab 7 — Deret Waktu dan Model Sekuensial

| `[n]` | Key `refs.bib` | Jenis | DOI / ISBN / arXiv | Status |
|---|---|---|---|---|
| [1] | `goodfellow2016deep` | Buku | (MIT Press) | ✅ |
| [2] | `hyndman2021fpp3` | Buku (open) | otexts.com/fpp3 | ✅ |
| [3] | `elman1990finding` | Artikel | 10.1207/s15516709cog1402_1 | ✅ |
| [4] | `hochreiter1997lstm` | Artikel | 10.1162/neco.1997.9.8.1735 | ✅ |
| [5] | `cho2014gru` | Artikel (arXiv) | arXiv:1406.1078 | ✅ |
| [6] | `sutskever2014seq2seq` | Artikel (NeurIPS) | arXiv:1409.3215 | ✅ |
| [7] | `abadi2016tensorflow` | Software/arXiv | arXiv:1603.04467 | ✅ |

### Bab 8 — Studi Kasus Pasang Surut Indonesia (Contoh Cilacap)

| `[n]` | Key `refs.bib` | Jenis | DOI / ISBN / arXiv | Status |
|---|---|---|---|---|
| [1] | `rob_kalbar` | Web (BIG) | URL tides.big.go.id | ✅ |
| [2] | `ioc_sealevel` | Web (UNESCO/IOC) | URL ioc-sealevelmonitoring.org | ✅ |
| [3] | `psmsl` | Web (PSMSL) | URL psmsl.org | ✅ |
| [4] | `uhslc_rqds` | Web (UHSLC) | URL uhslc.soest.hawaii.edu/data/ | ✅ |
| [5] | `pugh2014sealevel` | Buku | (Cambridge Univ. Press) | ✅ |
| [6] | `goodfellow2016deep` | Buku | (MIT Press) | ✅ |
| [7] | `hyndman2021fpp3` | Buku (open) | otexts.com/fpp3 | ✅ |
| [8] | `wmo2018verification` | Tech report (WMO) | (pdf WMO) | ✅ |
| [9] | `holgate2013psmsl` | Artikel (JCR) | 10.2112/JCOASTRES-D-12-00175.1 | ✅ |
| [10] | `abadi2016tensorflow` | Software/arXiv | arXiv:1603.04467 | ✅ |

### Bab 9 — Studi Kasus Curah Hujan BMKG

| `[n]` | Key `refs.bib` | Jenis | DOI / ISBN / arXiv | Status |
|---|---|---|---|---|
| [1] | `bmkg_dataonline` | Web (BMKG) | URL dataonline.bmkg.go.id | ✅ |
| [2] | `c3s_era5` | Web (C3S) | URL cds.climate.copernicus.eu | ✅ |
| [3] | `hersbach2020era5` | Artikel | 10.1002/qj.3803 | ✅ |
| [4] | `wheeler2004rmm` | Artikel | 10.1175/1520-0493(2004)132<1917:AARMMI>2.0.CO;2 | ✅ |
| [5] | `wolter1998mei` | Prosiding | (17th Climate Diagnostics) | ✅ |
| [6] | `wmo2018verification` | Tech report (WMO) | (pdf WMO) | ✅ |
| [7] | `hyndman2021fpp3` | Buku (open) | otexts.com/fpp3 | ✅ |
| [8] | `abadi2016tensorflow` | Software/arXiv | arXiv:1603.04467 | ✅ |

### Bab 10 — Operasional & Arah Riset

| `[n]` | Key `refs.bib` | Jenis | DOI / ISBN / arXiv | Status |
|---|---|---|---|---|
| [1] | `goodfellow2016deep` | Buku | (MIT Press) | ✅ |
| [2] | `hyndman2021fpp3` | Buku (open) | otexts.com/fpp3 | ✅ |
| [3] | `wmo2018verification` | Tech report (WMO) | (pdf WMO) | ✅ |
| [4] | `gal2016dropout` | Artikel (ICML) | arXiv:1506.02142 | ✅ |
| [5] | `lundberg2017shap` | Artikel (NeurIPS) | arXiv:1705.07874 | ✅ |
| [6] | `rasp2020weatherbench` | Artikel | 10.1029/2020MS002203 | ✅ |
| [7] | `reichstein2019deep` | Artikel | 10.1038/s41586-019-0912-1 | ✅ |
| [8] | `abadi2016tensorflow` | Software/arXiv | arXiv:1603.04467 | ✅ |

---

## Catatan Pemeliharaan

- **Saat menambah aset/sitasi baru di `master.md`:** perbarui file ini segera, sebelum
  menutup bab tersebut.
- **Nomor aset & sitasi mengikuti `master.md`** (sumber kebenaran). Jangan menomori ulang
  hanya di register.
- `CHECKLIST-EVALUASI.md` Fase 2 bagian C (sitasi) & E (build) menggunakan register ini
  sebagai input ke cek otomatis.