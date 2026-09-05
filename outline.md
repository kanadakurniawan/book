# Outline Buku — *Pengantar Deep Learning untuk Meteorologi*

> **Sumber kebenaran (single source of truth) untuk rencana & detil isi buku.**
> Dokumen operasional: dipakai saat menulis tiap bab (`master.md`), merencanakan
> konversi artikel blog, dan evaluasi pacing rilis.
> Strategi brand & metrik ada di `brand-strategy.md` (internal, git-ignored) — dokumen ini
> **ter-commit** sebagai blueprint publik.

- Estimasi total: **±195–260 halaman (du ~220)** pada format PDF/A5.
- Struktur: 4 bagian, 10 bab + front/back matter.
- Pola per bab: **1 bab ≈ 1 artikel blog ≈ 1 notebook Colab.**
- **1 buku = 1 DOI Zenodo** (PDF+DOCX utuh), versi bertambah tiap milestone rilis:
  v1.0 (Bab 1–5) → v1.1 (Bab 6–9) → v2.0 (lengkap 10 bab).
- Setiap bab berdiri sendiri; sidebar "Prasyarat: Bab …" memetakan urutan baca; notasi &
  terminologi seragam di seluruh buku (istilah Indonesia + Inggris di pemunculan pertama).

### Target Volume per Bab (agar layak sebagai "bab buku")

Bab 1–2 saat ini ~1.100–1.400 kata (terlalu tipis). Target tambahan saat menulis ulang
agar satu buku utuh terasa seimbang & pantas di Zenodo:

| Bab | Target kata isi | Catatan |
|---|---|---|
| 1 | 3.000–3.500 | Pengantar + setup + peta aplikasi diperluas |
| 2 | 3.500–4.000 | Tambah turunan loss, contoh numerik lebih detail |
| 3 | 3.000–3.500 | Tambah contoh confusion matrix + trade-off threshold |
| 4 | 3.500–4.000 | Kalkulasi backprop tertulis (langkah demi langkah) |
| 5 | 3.500–4.000 | Tambah studi mini sample imbalance |
| 6 | 3.500–4.000 | Tambah studi data BMKG/ERA5 nyata |
| 7 | 4.000–4.500 | Tambah jadwal multi-step & perbandingan univariate/multivariate |
| 8 | 4.000–4.500 | Perdalam framing analisis harmonik vs ML |
| 9 | 4.000–4.500 | Table verifikasi kategori + interpretasi |
| 10 | 3.000–3.500 | Operasional + etika + arah riset |

Total kata: ~36.000–40.000 kata isi (≈ 170–200 hal A5 PDF; dengan gambar diperkirakan
±220 halaman, cocok rentang estimasi). Laju target: **1–2 bab/bulan**.

---

## Tujuan Pembelajaran Buku (setelah membaca seluruh buku)

Setelah menyelesaikan buku ini, pembaca diharapkan mampu:

1. **Menjelaskan** posisi deep learning dalam AI/ML dan menilai secara kritis kapan
   deep learning layak digunakan untuk masalah meteorologi dibanding baseline statistik
   (Bab 1, 5, 7, 10).
2. **Membangun** model deep learning (regresi dan klasifikasi) dengan TensorFlow/Keras
   untuk data deret waktu meteorologi Indonesia, dari persiapan data hingga evaluasi
   (Bab 2–7).
3. **Menerapkan** prosedur data yang benar: pembersihan, imputasi, fitur, normalisasi,
   dan split berbasis waktu yang mencegah *leakage* (Bab 5–6).
4. **Mengevaluasi** model dengan metrik yang tepat untuk konteks operasional — MAE/RMSE
   untuk regresi, CSI/FAR/POD untuk kejadian ekstrem — serta membandingkannya dengan
   baseline (Bab 2, 5, 8–9).
5. **Mereproduksi** studi kasus end-to-end pasang surut Kapuas dan curah hujan BMKG,
   menginterpretasi hasil, dan mengenali batas model (Bab 8–9).
6. **Mengomunikasikan** hasil model kepada praktisi/penentu kebijakan secara jujur,
   termasuk keterbatasan, ketidakpastian, dan implikasi etika (Bab 10).
7. **Menindaklanjuti** pembelajaran ke arah riset lanjutan (CNN, nowcasting, downscaling,
   generative) dengan peta jalan yang tersedia (Bab 10).

> Tujuan per bab dijabarkan di masing-masing bagian bab; latihan pada tiap bab dirancang
> untuk menguji tujuan tersebut (constructive alignment).

---

## Bagian I — Fondasi (Bab 1–5)

### Bab 1 — Pengantar: Deep Learning untuk Meteorologi
`book/manuscripts/ch-01-pengantar-deep-learning-meteorologi/master.md` · ±15–20 hal

- **Tujuan pembelajaran:** setelah bab ini, pembaca mampu:
  1. Membedakan AI, machine learning, dan deep learning beserta contoh aplikasinya.
  2. Mempetakan aplikasi DL meteorologi ke bab yang relevan dan membedakan mana yang
     dibahas buku vs literatur lanjut.
  3. Menilai kapan deep learning layak dipakai (vs baseline statistik) berdasarkan ukuran
     data, non-linearitas, dan konteks operasional.
  4. Menyiapkan lingkungan kerja Google Colab + TensorFlow/Keras dan membuat tensor
     pertama dari contoh data cuaca mini.
- **Isi:**
  1. ML vs DL — dan di mana DL berada dalam spektrum (regresi, klasifikasi, dsb).
  2. Mengapa DL relevan *sekarang*: ketersediaan data besar, GPU/Colab gratis, tooling matang.
  3. **Peta aplikasi meteo** — Buku vs diarahkan ke sumber lain:

     | Aplikasi | Di mana di buku | Jika tidak |
     |---|---|---|
     | Prediksi deret waktu (pasang surut, hujan) | **Bab 6–9** | — |
     | Klasifikasi kejadian (hujan/tidak, level bahaya) | **Bab 3, 5, 9** | — |
     | Nowcasting / radar-satelit | — | Bab 10 (arah riset), literatur lanjut |
     | Downscaling & data spasial | — | Bab 10 (arah riset) |
     | Imputasi data hilang | Bab 6 (dasar) | Bab 10 (generative) |

  4. **Kapan DL layak vs model statistik** (regresi, ARIMA): ukuran data, kompleksitas pola,
     biaya + teaser baseline di Bab 7 ("DL harus mengalahkan baseline").
  5. Setup lingkungan: Google Colab, TensorFlow/Keras, verifikasi instalasi.
  6. Pengenalan tensor 0D–3D dengan contoh data cuaca mini (skalar suhu, vektor, matriks stasiun).
- **Notebook:** `00_fondasi_tensorflow`.
- **Gaya:** perkenalan hangat + formal; banyak diagram peta aplikasi.
- **SEO:** "apa itu deep learning", "deep learning untuk meteorologi",
  "pengantar deep learning bahasa Indonesia".
- **Blog:** 2 artikel (konsep; panduan setup Colab/TensorFlow).
- **Referensi kunci:** LeCun et al. (2015), Reichstein et al. (2019), Goodfellow et al. (2016).

---

### Bab 2 — Regresi: Perceptron dan Jaringan Saraf untuk Prediksi Besaran
`book/manuscripts/ch-02-regresi-neural-network/master.md` · ±20–25 hal

- **Tujuan pembelajaran:** setelah bab ini, pembaca mampu:
  1. Membangun model regresi neural network (perceptron/MLP) untuk prediksi besaran
     meteorologi.
  2. Menjelaskan peran bobot, bias, dan fungsi aktivasi (termasuk ReLU) serta kapan
     non-linearitas diperlukan.
  3. Menerapkan mini-kasus pasang surut: windowing, baseline persistence, dan perbandingan
     MAE antara model neural vs baseline.
  4. Memilih antara MAE dan MSE berdasarkan sifat data dan tujuan, serta membagi data
     deret waktu secara kronologis yang mencegah leakage.
- **Isi:**
  1. Framing masalah meteo → regresi (suhu, jumlah hujan, tinggi pasang).
  2. Anatomi neuron: bobot, bias, fungsi aktivasi.
  3. 1 neuron linear = regresi linear → MLP + **ReLU muncul dari kebutuhan non-linearitas**.
  4. **Mini-kasus pasang surut sederhana** (identitas "data laut Indonesia" hadir sejak bab ini).
  5. MAE vs MSE — dan skala ekstrem data meteo.
  6. Adam & learning rate (pengenalan; mendalam di Bab 4).
  7. Motif split train/val/test + kenapa urutan waktu krusial di meteo.
- **Notebook:** `01_regresi_pasang_surut`.
- **Latihan:** 4–5 soal + proyek mini regresi suhu stasiun lokal.
- **SEO:** "regresi neural network", "prediksi suhu machine learning",
  "deep learning regresi Indonesia".
- **Blog:** 1–2 artikel.

---

### Bab 3 — Klasifikasi: Mengenali Kategori Fenomena
`book/manuscripts/ch-03-klasifikasi-neural-network/master.md` · ±18–22 hal

- **Tujuan pembelajaran:** setelah bab ini, pembaca mampu:
  1. Membangun model klasifikasi biner dan multi-kelas (hujan/tidak, level bahaya) dengan
     TensorFlow/Keras.
  2. Menjelaskan peran sigmoid/softmax dan binary/categorical cross-entropy.
  3. Mendiagnosis *class imbalance* dan memilih metrik yang tepat (precision, recall, F1,
     pengenalan CSI/FAR) — bukan hanya akurasi.
  4. Menerapkan trade-off threshold ala praktisi peramalan untuk fenomena langka.
- **Isi:**
  1. Klasifikasi hujan/tidak hujan; kategori level bahaya (waspada–siaga–awas).
  2. Sigmoid & softmax — hubungan dengan probabilitas.
  3. Binary/categorical cross-entropy.
  4. **Class imbalance** — mengapa akurasi menipu untuk hujan lebat/jarang.
  5. Precision/recall/F1, confusion matrix, pengenalan **CSI/FAR** (digunakan penuh di Bab 5).
  6. Trade-off threshold ala praktisi peramalan.
- **Notebook:** `02_klasifikasi_hujan`.
- **Latihan:** proyek klasifikasi stasiun data; bandingkan threshold.
- **SEO:** "klasifikasi machine learning", "prediksi hujan tidak hujan",
  "precision recall klasifikasi".
- **Blog:** 1–2 artikel.

---

### Bab 4 — Backpropagation, Optimasi dan Pelatihan
`book/manuscripts/ch-04-backpropagation-optimasi/master.md` · ±18–22 hal

- **Tujuan pembelajaran:** setelah bab ini, pembaca mampu:
  1. Menjelaskan mekanisme gradient descent dan backpropagation (aturan rantai) secara
     intuitif.
  2. Menganalisis peran fungsi aktivasi dari sisi gradien (ReLU vs sigmoid/tanh) dan
     mengenali vanishing gradient.
  3. Menerapkan tuning hyperparameter (learning rate, batch size, epochs) dan callback
     (early stopping, ModelCheckpoint, ReduceLROnPlateau).
  4. Membaca learning curve untuk mendiagnosa underfit/overfit sebagai transisi ke Bab 5.
- **Isi:**
  1. Intuisi gradient descent; backpropagation via aturan rantai.
  2. **Fungsi aktivasi ditinjau dari sisi gradien** — ReLU vs sigmoid/tanh, vanishing gradient
     (rumah pembahasan lengkap "fungsi aktivasi" yang semula bab mandiri).
  3. Learning rate & scheduler; batch size; epochs.
  4. SGD vs Adam; callback: early stopping, ModelCheckpoint, ReduceLROnPlateau.
  5. Membaca learning curve → transisi ke Bab 5.
- **Notebook:** `03_optimasi_callbacks`.
- **Latihan:** tuning LR/batch pada kasus regresi Bab 2.
- **SEO:** "backpropagation", "cara kerja neural network", "learning rate neural network".
- **Blog:** 1–2 artikel.

---

### Bab 5 — Overfitting, Regularisasi dan Evaluasi untuk Data Iklim
`book/manuscripts/ch-05-overfitting-regularisasi-evaluasi/master.md` · ±18–22 hal

- **Tujuan pembelajaran:** setelah bab ini, pembaca mampu:
  1. Mendiagnosa underfit/overfit melalui learning curve dan konsep bias-variance.
  2. Menerapkan regularisasi (L2, dropout, early stopping) untuk mencegah overfit.
  3. Memilih metrik operasional yang tepat (MAE/RMSE/R²/Willmott/KGE dan CSI/FAR/POD/TS)
     sesuai tujuan.
  4. Menerapkan cross-validation deret waktu yang benar (walk-forward/blocked) dan
     mencegah leakage.
- **Isi:**
  1. Bias-variance, under/overfit via learning curve.
  2. L2, dropout, early stopping.
  3. **Metrik domain:** MAE, RMSE, R², Willmott, KGE (hidrologi) + tabel *kapan pakai metrik apa*.
  4. **CSI/FAR/POD/TS** untuk kejadian ekstrem (praktik favorit peramal cuaca).
  5. Cross-validation deret waktu yang benar: **walk-forward/blocked** (bukan k-fold acak),
     mencegah leakage.
- **Notebook:** `04_metrik_walkforward` (pipeline dipakai ulang di Bab 8–9).
- **Latihan:** evaluasi model Bab 3 dengan CSI/FAR di berbagai threshold.
- **SEO:** "overfitting", "evaluasi model prediksi cuaca", "cross validation time series".
- **Blog:** 1 artikel inti + tabel cepat.

---

## Bagian II — Data & Model Sekuensial (Bab 6–7)

### Bab 6 — Data Meteorologi: Sumber, Kualitas dan Persiapan
`book/manuscripts/ch-06-data-meteorologi/master.md` · ±18–24 hal

- **Tujuan pembelajaran:** setelah bab ini, pembaca mampu:
  1. Mengambil dan menghubungkan data meteorologi Indonesia (BMKG, ERA5, pasang surut)
     beserta lisensi dan batasannya.
  2. Membaca/menulis format CSV, NetCDF, GRIB dan menangani nilai hilang, outlier, serta
     imputasi dasar.
  3. Melakukan eksplorasi (dekomposisi musiman, distribusi, korelasi) dan feature
     engineering (lag, musiman, ENSO/MJO).
  4. Menerapkan normalisasi (fit pada train) dan split berbasis waktu anti-leakage.
- **Isi:**
  1. Sumber data: **stasiun BMKG**, reanalysis **ERA5** (Copernicus), CMIP6, **pasang surut
     (PSMSL/IOC/BIG)**, satelit. Lisensi & batasan akses.
  2. Format: CSV, NetCDF, GRIB; tooling xarray, netCDF4.
  3. Kualitas: nilai hilang, outlier, imputasi dasar.
  4. Eksplorasi: dekomposisi musiman, distribusi ekor kanan hujan, korelasi silang.
  5. Feature engineering: lag, indikator musiman, **ENSO/MJO sebagai fitur contoh**.
  6. Normalisasi (fit **hanya** pada train); split berbasis waktu + anti-leakage.
- **Notebook:** `05_persiapan_data` (hujan harian + fitur ERA5).
- **Latihan:** bangun dataset sendiri dari BMKG/ERA5 untuk stasiun pilihan.
- **SEO:** "data cuaca untuk machine learning", "dataset meteorologi Indonesia",
  "tutorial ERA5", "data curah hujan BMKG".
- **Blog:** 2–3 artikel (tutorial data BMKG; tutorial ERA5; checklist kualitas).

---

### Bab 7 — Time Series dan Model Sekuensial: RNN, LSTM, GRU
`book/manuscripts/ch-07-time-series-lstm-gru/master.md` · ±20–28 hal

- **Tujuan pembelajaran:** setelah bab ini, pembaca mampu:
  1. Menyusun deret waktu menjadi contoh-window untuk prediksi satu dan multi-langkah.
  2. Membandingkan LSTM/GRU dengan baseline (persistence, mean, AR(p)) secara jujur.
  3. Menjelaskan intuisi RNN → LSTM → GRU (gate ingatan/lupa) dan keterbatasannya.
  4. Memilih arsitektur input multivariate dan strategi multi-step (recursive/direct/seq2seq).
- **Isi:**
  1. Windows & horizon: satu vs multi-langkah.
  2. **Baseline dulu:** persistence, mean, AR(p) — DL harus mengalahkan baseline.
  3. RNN & keterbatasannya → LSTM (intuisi gate = "pintu ingatan & lupa") → GRU (ringkas).
  4. Arsitektur praktis: input shape, univariate vs multivariate.
  5. Multi-step: recursive vs direct vs seq2seq (pengenalan).
  6. Evaluasi + plotting forecast vs aktual; catatan training.
- **Notebook:** `06_lstm_gru` (menyambung Bab 8).
- **Latihan:** bandingkan baseline vs LSTM pada data Bab 6.
- **SEO:** "LSTM time series", "prediksi deret waktu LSTM", "GRU vs LSTM".
- **Blog:** 2 artikel.

---

## Bagian III — Studi Kasus (Bab 8–9)

### Bab 8 — Studi Kasus: Prediksi Pasang Surut di Perairan Indonesia (Contoh Cilacap)
`book/manuscripts/ch-08-studi-kasus-pasang-surut-kapuas/master.md` · ±20–28 hal

- **Tujuan pembelajaran:** setelah bab ini, pembaca mampu:
  1. Menjalankan proyek end-to-end prediksi pasang surut dari data terbuka
     (IOC/UHSLC/PSMSL), dengan Cilacap sebagai contoh reproducible.
  2. Menerapkan pipeline Bab 7 (baseline persistence vs MLP vs LSTM/GRU) dengan
     walk-forward.
  3. Mengevaluasi MAE/RMSE terhadap toleransi tinggi pasang dan memplot prediksi
     1–7 hari.
  4. Menjelaskan framing jujur: ML untuk prakiraan cepat & pengisian gap, bukan
     klaim riset baru.
  5. Mengenali keterbatasan ketika lokasi studi (mis. Pontianak/Kalimantan) tidak
     memiliki tide gauge terbuka, dan memetakan strategi fallback (proksi,
     FES2014/GOT4.10, kerja sama BIG/BRIN).
- **Isi:**
  1. Konteks: banjir rob pesisir Indonesia (Jakarta/Semarang/Cilacap/Pontianak);
     jenis pasang (semi-diurnal/diurnal/campuran); tipe pasut Indonesia.
  2. **Pemilihan station:** Cilacap (GLOSS #291) sebagai contoh reproducible —
     mengapa, keterbatasan untuk lokasi tanpa station terbuka.
  3. **Sumber data terbuka:** IOC, UHSLC, PSMSL, BIG; tabel station Indonesia yang
     datanya dapat diunduh; lisensi & atribusi.
  4. Pipeline Bab 7: baseline persistence vs MLP vs LSTM/GRU; walk-forward 4 blok.
  5. Evaluasi MAE/RMSE vs toleransi tinggi pasang; plot prediksi 1–7 hari;
     diskusi batas model & keterbatasan data.
  6. **Framing jujur:** analisis harmonik untuk penjelasan, ML untuk prakiraan
     cepat & pengisian gap data — bukan klaim riset baru.
- **Data & reproduksibilitas:**
  - **Sample CSV** di-commit: `data/sample/cili_1y_hourly.csv` (1 tahun hourly
    sintetik realistis untuk out-of-the-box notebook).
  - **Skrip unduh** (`scripts/download_ioc.py`) untuk IOC real-time + UHSLC
    ERDDAP + PSMSL metadata.
  - Data riil hasil unduh masuk `data/raw/` (di-`.gitignore`); snapshot Zenodo
    untuk DOI data kasus (rujuk `bookDOI`).
- **Latihan:** prediksi station lain (mis. Ambon/Bitung via IOC); bandingkan
  tipe pasang; simulasi gap; laporan 1 halaman.
- **SEO:** "prediksi pasang surut LSTM", "pasang surut Cilacap GLOSS",
  "pasang surut Indonesia machine learning", "banjir rob deep learning".
- **Blog:** 3–4 artikel (konteks pasang surut Indonesia; eksperimen Cilacap;
  insight keterbukaan data) + **video YouTube** (Fase II).

---

### Bab 9 — Studi Kasus: Prediksi Curah Hujan Stasiun BMKG
`book/manuscripts/ch-09-studi-kasus-curah-hujan-bmkg/master.md` · ±22–30 hal

- **Tujuan pembelajaran:** setelah bab ini, pembaca mampu:
  1. Membangun prediktor hujan stasiun BMKG (regresi jumlah hujan + klasifikasi intensitas).
  2. Menerapkan verifikasi operasional dengan CSI/FAR/POD dan trade-off threshold.
  3. Membandingkan walk-forward vs baseline (persistence, klimatologi, ARIMA singkat).
  4. Melakukan interpretasi awal (permutation importance/SHAP) dan menyusun tabel verifikasi
     per kategori.
- **Isi:**
  1. Konteks layanan (BBMKG, peringatan dini) tanpa sensasionalisme.
  2. Data hujan harian stasiun + fitur regional ERA5 + musiman.
  3. Dua lintasan:
     - Regresi jumlah hujan (MAE + metrik domain).
     - Klasifikasi hujan/tidak & kategori intensitas dengan **CSI/FAR/POD + trade-off threshold**.
  4. Walk-forward vs baseline (persistence, klimatologi, ARIMA singkat).
  5. Interpretasi awal (permutation importance/SHAP) → transisi ke Bab 10.
  6. Tabel verifikasi per kategori intensitas.
- **Reproduksibilitas:** sumber data, lisensi, pipeline, seed, versi/**DOI**.
- **Latihan:** model untuk stasiun pola berbeda (Indonesia timur vs barat).
- **SEO:** "prediksi curah hujan machine learning", "prediksi hujan LSTM",
  "machine learning BMKG".
- **Blog:** 3–4 artikel + **video YouTube** (Fase II).

---

## Bagian IV — Operasional & Arah Riset (Bab 10)

### Bab 10 — Dari Riset ke Praktik: Operasional, Interpretasi, dan Arah ke Depan
`book/manuscripts/ch-10-operasional-arah-riset/master.md` · ±16–22 hal

- **Tujuan pembelajaran:** setelah bab ini, pembaca mampu:
  1. Merancang monitoring drift dan strategi retraining/kalibrasi untuk model operasional.
  2. Mengkuantifikasi ketidakpastian prediksi (interval/quantile, ensembel multi-seed).
  3. Menginterpretasi model (SHAP) dan mengaitkannya dengan pengetahuan atmosfer.
  4. Menjelaskan keterbatasan & etika penggunaan DL di institusi (anti-overhype, disclaimer).
  5. Menyusun peta jalan belajar lanjut (CNN, nowcasting, downscaling, generative).
- **Isi:**
  1. **Monitoring drift** — atmosfer non-stasioner, model "rusak" seiring waktu;
     strategi retraining/kalibrasi ulang.
  2. Ketidakpastian: interval/quantile, ensembel multi-seed.
  3. Interpretasi: SHAP & tautan ke pengetahuan atmosfer (fitur yang "masuk akal"
     menaikkan kepercayaan praktisi).
  4. **Keterbatasan & etika:** framing pengenalan, hindari overhype, disclaimer
     (sesuai Risk Management umbrella).
  5. **Arah riset:** CNN & data spasial (nowcasting radar/satelit, downscaling),
     transfer learning (singkat), **diffusion/generative untuk imputasi data & skenario iklim**.
  6. Komunitas & peta belajar lanjut.
- **Blog:** artikel "deep learning operasional BMKG" + peta belajar lanjut.
- **SEO:** "deep learning operasional cuaca", "arah riset AI meteorologi",
  "nowcasting machine learning".

---

## Front & Back Matter (±15–25 hal)

- **Front matter:** Halaman judul, Prakata, *Cara Memakai Buku* (peta baca) — sumbernya
  `front-matter/cara-memakai-buku.md` (di luar `manuscripts/`, **tidak** dirilis ke Zenodo),
  Glosarium & Notasi.
- **Back matter:** Daftar Pustaka (IEEE, sertakan DOI), Indeks, Daftar Dataset & Sumber,
  Daftar Notebook & DOI.

---

## Pacing Rilis (sinkron dengan `brand-strategy.md`)

| Fase | Rilis |
|---|---|
| I (0–6 bln, Trust) | Bab 1–5 (fondasi + evaluasi) + Zenodo + blog; mini-kasus pasang surut hadir sejak Bab 2; validasi 3–5 pembaca; SEO + analytics |
| II (6–12 bln, Reach) | Bab 6–7 (data + time series) → Bab 8–9 (dua kasus flagship) + video YouTube per bab kasus; notebook disematkan di blog |
| III (1+ thn, Scale) | Bab 10 + bundel final v1.x; bahan kursus Udemy dari Bab 6–9; update DOI saat revisi |

---

## Aturan Konsistensi Global

1. **Template seragam tiap bab:** Tujuan Pembelajaran (3–5 butir aksi) → Pembukaan masalah →
   Isi/konsep → Kode/notebook → Ringkasan kunci → Latihan → Referensi (IEEE, sertakan DOI)
   → Keyword SEO. Latihan tiap bab dirancang untuk menguji Tujuan Pembelajaran
   (*constructive alignment*); rujukan tujuan buku di bagian "Tujuan Pembelajaran Buku".
2. **Notasi & glosarium satu sumber:** istilah Indonesia + Inggris di pemunculan pertama
   (mis. "fungsi aktivasi (*activation function*)"); istilah sama di blog/buku/YouTube.
3. **Setiap bab berdiri sendiri** (sidebar "Prasyarat: Bab …"), tapi satu narasi & notasi.
4. **Reproduksibilitas:** seed tetap, versi TensorFlow, snapshot data di Zenodo,
   1 notebook per bab + template notebook kasus.
5. **Tone:** formal-hangat sesuai umbrella §3.1b; hindari hiperbola ("materi pengenalan",
   bukan klaim riset baru).
6. **Framing jujur di kasus:** DL dibandingkan dengan baseline (persistence/ARIMA/harmonik);
   hasil dilaporkan apa adanya.
7. **Istilah asing dicetak miring (PUEBI §18):** kata, frasa, atau istilah bahasa Inggris/
   asing yang **belum diserap** ke dalam bahasa Indonesia ditulis dengan *huruf miring*
   (Markdown: `*…*`). Berlaku untuk badan teks dan daftar. Acuan utama:
   - **Wajib miring** (istilah asing yang belum diserap KBBI): *baseline*, *persistence*,
     *climatology*, *overhype*, *overfit*, *underfit*, *overfitting*, *underfitting*,
     *trade-off*, *end-to-end*, *time series*, *sequence model*, *sequence-to-sequence*,
     *neural network*, *deep learning*, *machine learning*, *feature*, *loss function*,
     *hyperparameter*, *epoch*, *batch size*, *learning rate*, *gradient descent*,
     *backpropagation*, *callbacks*, *early stopping*, *dropout*, *seed*, *nowcasting*,
     *downscaling*, *generative*, *chatbot*, *pipeline*, *state of the art*, *sophisticated*,
     *fold* (pada k-fold), *leakage*, *walk-forward*, *skill score*, dst.
   - **TIDAK miring** (nama diri / merek / istilah KBBI yang sudah diserap): TensorFlow,
     PyTorch, Keras, NumPy, Pandas, scikit-learn, xarray, Google Colab, Python, GitHub,
     Zenodo, DOI, ISBN, arXiv, GPU, CPU, TPU, BMKG, ERA5, ERA5-Land, BMKG, WMO, IEEE,
     Colab, Notebook, Internet. Akronim/singkatan organisasi juga tidak miring.
   - **Penulisan pertama** untuk istilah yang punya padanan Indonesia: pakai pola
     "padanan Indonesia (*istilah Inggris*)" (mis. "tolok ukur (*baseline*)") dan
     setelahnya konsisten: hanya pakai padanan Indonesia ATAU hanya istilah Inggris
     miring, jangan dicampur dalam satu paragraf.
   - **Di dalam blok kode, label plot, dan identifier program** TIDAK diubah menjadi
     miring (kode adalah teks verbatim). Konsistensi miring hanya untuk badan teks
     naratif dan daftar.
   - **Di REGISTER.md dan outline.md** kode rujukan internal seperti "§1.6" tetap
     dipakai apa adanya; REGISTER/outline bukan badan teks buku.

---

## Kriteria Sitasi (wajib untuk semua bab)

Standar sitasi seluruh bab, selaras dengan aturan IEEE di umbrella (§3.1b) dan prinsip
"etika sitasi" (selalu kutip sumber, sertakan DOI, periksa fakta).

### 1. Gaya & Format (aturan dasar)

- **Gaya:** IEEE, sitasi bernomor `[1]`, `[2]`, dst — muncul **berurutan sesuai kemunculan
  pertama** di teks.
- Format referensi di daftar akhir mengikuti IEEE: penulis → judul → sumber → volume/
  halaman → tahun → DOI.
- Setiap bab wajib punya berkas `refs.bib` yang **identik urutan dan isinya** dengan daftar
  `References` di `master.md` (agar output PDF via citeproc sama dengan versi Markdown/blog).

### 2. Verifikasi `refs.bib` & DOI (wajib lolos sebelum rilis)

- **Semua referensi harus benar-benar ada** — author, judul, jurnal/prosiding, volume,
  halaman, tahun dicek silang, bukan dihafal/tebak.
- **DOI wajib dicantumkan bila tersedia**, dan diuji lewat `doi.org/<doi>`; DOI yang salah
  dianggap cacat.
- URL (mis. dokumentasi TensorFlow) hanya bila tidak ada DOI, dan sertakan **tanggal akses**.

### 3. Jenis Sumber — hierarki prioritas

| Prioritas | Jenis | Catatan |
|---|---|---|
| 1 | Jurnal/prosiding *peer-reviewed*, buku teks klasik | Sumber utama klaim teknis |
| 2 | Buku teks DL/ML (Goodfellow, Bishop, Géron, Chollet, Nielsen) | Untuk definisi & derivasi inti |
| 3 | Dokumentasi resmi library (TensorFlow, Keras, xarray, Pandas) | Cantumkan versi API |
| 4 | Dataset & data (BMKG, ERA5/Copernicus, PSMSL/BIG) | Wajib: lisensi, identifikasi dataset, versi, cara akses |
| 5 | Preprint (arXiv/SSRN) | Boleh bila tak ada versi peer-reviewed; tandai "preprint" |
| 6 | Blog/artikel non-review, Wikipedia | **Hanya untuk konteks/lintasan**, bukan penguat klaim inti; usahakan diganti sumber primer |
| ✗ | Sumber sekunder tanpa kredibilitas, tautan mati, UGC tanpa verifikasi | Jangan |

### 4. Kriteria Keilmuan

- **Relevansi & keterkinian:** untuk topik yang cepat berubah (arsitektur DL, tooling),
  utamakan rilis ≤ 5–10 tahun; buku teks klasik boleh lama asal fondasi.
- **Kebenaran klaim:** setiap pernyataan substantif (definisi, angka, sifat matematis,
  klaim performa) harus bersitasi; data/kutipan harus bisa dilacak ke sumber.
- **Bahasa:** preferensikan sumber berbahasa Inggris (ilmiah). Sumber berbahasa Indonesia
  hanya dari institusi resmi (BMKG, BPP) atau jurnal nasional terindeks.
- **Konteks meteorologi:** preferensikan literatur domain (meteo/ocean/hidro) bila ada,
  misal paper time series/pasang surut/ML-cuaca; jangan hanya kutip literatur CS.

### 5. Aturan Penggunaan dalam Teks

- **Sitasi per klaim**, bukan per paragraf kabur — pembaca bisa menelusuri asal setiap
  pernyataan teknis.
- **Parafrase, bukan salin-tempel kata demi kata** (hindari plagiarisme); kutipan langsung
  pendek diberi tanda kutip.
- **Konsistensi lintas kanal:** istilah + sitasi yang sama di buku, blog, dan YouTube
  (minimal satu sitasi primer per artikel/video).
- **Konteks gambar & data:** setiap gambar/tabel yang diambil memberi atribusi + lisensi di
  keterangan; data disebutkan sumbernya, bukan hanya "BMKG" tanpa spesifik.
- **Self-citation:** kutip DOI buku sendiri (melalui `bookDOI` pada bab terkait)
  diperbolehkan secukupnya, tidak berlebihan, dan jangan jadi mayoritas referensi.

### 6. Jumlah & Struktur per Bab

- **Fondasi (Bab 1–5):** 6–12 referensi, utamanya buku teks + paper kunci.
- **Data & sekuensial (Bab 6–7):** 8–15, termasuk dokumentasi dataset/library.
- **Studi kasus & operasional (Bab 8–10):** 10–20, termasuk paper domain (pasang surut,
  ML cuaca, verifikasi) + data sumber.
- Hampir selalu akhiri referensi dengan DOI bila tersedia (ini pula nilai diferensiasi
  "DOI sebagai kredibilitas" di umbrella).

### 7. Kontrol Kualitas Final

- Jalankan cek sitasi: **semua `[n]` di teks harus ada di daftar akhir** dan sebaliknya;
  urutan incremental pertama-muncul.
- Uji output `node build/generate.mjs` → PDF; pastikan `refs.bib` tidak ada kunci dobel
  dan citeproc bebas error.

---

## Alur Kerja Penulisan & Evaluasi Internal

Urutan kerja untuk menyelesaikan buku menjadi satu dokumen final, **sebelum** DOI/ISBN,
GitHub publik, dan blog.

### Fase 1 — Menulis (Bab 1–10)

1. Tulis `manuscripts/ch-0N-<slug>/master.md` + `refs.bib` + `figures/` + `notebooks/`
   per bab, mengikuti **target volume di atas** (3.000–4.500 kata/bab).
2. **Commit lokal secara berkala** — minimal satu commit per bab (backup & riwayat).
   Repo bisa **private** selama development; publik dibuat saat siap rilis.
3. Notebook & figur disiapkan seiring bab (bukan ditunda ke akhir).

### Fase 2 — Evaluasi Internal (sebelum publikasi)

Checklist wajib lolos **seluruh bab** sebelum membuat DOI/ISBN:

**A. Isi & keilmuan**
- Semua klaim teknis disitasi; tidak ada pernyataan tanpa sumber untuk klaim substantif.
- Ketepatan istilah: istilah Indonesia + Inggris benar, konsisten di seluruh buku.
- Anti-overhype: semua hasil DL dibandingkan baseline; disclaimer "materi pengenalan" ada.

**B. Struktur & konsistensi**
- Template seragam tiap bab (Pembukaan → Tujuan → Isi → Kode → Ringkasan → Latihan →
  Referensi → SEO).
- Sidebar Prasyarat benar untuk tiap bab.
- Notasi & glosarium satu sumber (tidak ada istilah ganda).

**C. Sitasi**
- `[n]` di teks ≡ daftar References ≡ `refs.bib`; urutan incremental; DOI/ISBN/arXiv tercantum.
- Cek semua DOI via `doi.org/<doi>`; URL dengan tanggal akses.

**D. Kode & reproduksibilitas**
- Semua notebook dapat dieksekusi dari awal-akhir (Colab) tanpa error; seed tetap, versi TF
  tercatat.
- Data yang dipakai tersedia/di-zenodo (untuk kasus).

**E. Build & output**
- `node build/generate.mjs` menghasilkan PDF+DOCX tanpa error (perlu Pandoc+LaTeX pada
  mesin rilis).
- Total halaman realistis (target ±220); tidak ada bab yang terlalu kurus/gendut.

### Fase 3 — Penerbitan

Setelah evaluasi lolos:
1. **DOI Zenodo** — daftarkan buku utuh (PDF+DOCX `releases/`), konsep versi untuk revisi
   berikutnya; salin ke `bookDOI` semua bab.
2. **ISBN** — daftarkan buku (untuk versi cetak/standar internasional).
3. **GitHub publik** — buka repo (book + kode), rilis `v1.0.0` dengan tag.
4. **Blog** — posting **2 artikel/bulan** berbasis bab (dari buku final), lengkap dengan
   notebook & figur; sinkronkan via `node build/sync-to-blog.mjs`.

> Prinsip: tidak ada publikasi (Zenodo/ISBN/GitHub publik/blog) sebelum **semua 10 bab**
> selesai dan lolos evaluasi internal Fase 2.

---

## Catatan Revisi

- **Dari outline awal:** "Fungsi aktivasi" (bab mandiri) dibubarkan → muncul sebagai kebutuhan
  aplikasi di Bab 2–4; evaluasi domain ditambah Bab 5 agar kredibel di kalangan praktisi;
  data diperdalam Bab 6; kasus Kapuas & curah hujan jadi flagship setelah LSTM (Bab 8–9);
  operasional + arah riset digabung Bab 10.
- **Penyebab:** 10 iterasi review internal (audit struktur, learning-by-problem, framing tugas
  ML, evaluasi domain, adopsi pola code-first ala Bourke, data & reproduksibilitas, bobot &
  urutan kasus, sisi operasional, SEO & blog mapping, polish konsistensi).
- **Revisi studi kasus pasang surut (Bab 8):** studi kasus awal "Pontianak/Kapuas" diganti
  ke "Cilacap" (GLOSS #291, Indonesia) karena tidak ada tide gauge terbuka di Kalimantan.
  Tambahan: tabel station Indonesia di sumber terbuka, skrip `download_ioc.py` (IOC + UHSLC +
  PSMSL), sample CSV sintetik 1 tahun hourly untuk out-of-the-box notebook, dan keterbukaan
  eksplisit tentang keterbatasan untuk lokasi tanpa station terbuka (strategi fallback ke
  proksi terdekat atau model global FES2014/GOT4.10).