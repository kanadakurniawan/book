# Outline Buku — *Pengantar Deep Learning untuk Meteorologi*

> **Sumber kebenaran (single source of truth) untuk rencana & detil isi buku.**
> Dokumen operasional: dipakai saat menulis tiap bab (`master.md`), merencanakan
> konversi artikel blog, dan evaluasi pacing rilis.
> Strategi brand & metrik ada di `brand-strategy.md` (internal, git-ignored) — dokumen ini
> **ter-commit** sebagai blueprint publik.

- Estimasi total: **±195–260 halaman (du ~220)** pada format PDF/A5.
- Struktur: 4 bagian, 10 bab + front/back matter.
- Pola per bab: **1 bab ≈ 1 artikel blog ≈ 1 notebook Colab (≈1 DOI Zenodo)**.
- Setiap bab berdiri sendiri; sidebar "Prasyarat: Bab …" memetakan urutan baca; notasi &
  terminologi seragam di seluruh buku (istilah Indonesia + Inggris di pemunculan pertama).

---

## Bagian I — Fondasi (Bab 1–5)

### Bab 1 — Pengantar: Deep Learning untuk Meteorologi
`book/manuscripts/ch-01-pengantar-deep-learning-meteorologi/master.md` · ±15–20 hal

- **Tujuan bab:** memahami posisi DL dalam ML & kebumian; setup lingkungan kerja; memahami
  peta jalan buku.
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
- **Blog:** 2 artikel (konsep; panduan setup Colab/TensorFlow). DOI bab.
- **Referensi kunci:** LeCun et al. (2015), Reichstein et al. (2019), Goodfellow et al. (2016).

---

### Bab 2 — Regresi: Perceptron dan Jaringan Saraf untuk Prediksi Besaran
`book/manuscripts/ch-02-regresi-neural-network/master.md` · ±20–25 hal

- **Tujuan bab:** membangun model regresi pertama; memahami peran fungsi aktivasi; mengukur error.
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

- **Tujuan bab:** model klasifikasi biner & multi-kelas; memahami sigmoid/softmax; metrik untuk
  fenomena langka.
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

- **Tujuan bab:** memahami cara model belajar; skill praktis tuning hyperparameter.
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

- **Tujuan bab:** mendiagnosa & mencegah overfit; memilih metrik operasional yang benar.
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

- **Tujuan bab:** mengambil & membersihkan data meteo Indonesia; membangun dataset ML yang valid.
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

- **Tujuan bab:** menyusun deret waktu jadi data ML; membangun & membandingkan LSTM/GRU secara jujur.
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

### Bab 8 — Studi Kasus: Prediksi Pasang Surut di Perairan Kapuas (Pontianak)
`book/manuscripts/ch-08-studi-kasus-pasang-surut-kapuas/master.md` · ±20–28 hal

- **Tujuan bab:** proyek end-to-end pertama dari data Indonesia nyata; kredibilitas
  "dari praktisi untuk praktisi".
- **Isi:**
  1. Konteks: mengapa Kapuas/Pontianak (kota rendah berawa, banjir rob, interaksi pasang–debit
     sungai); jenis pasang (semi-diurnal/diurnal/campuran).
  2. **Framing jujur:** analisis harmonik untuk penjelasan, ML untuk prakiraan cepat &
     pengisian gap data — bukan klaim riset baru.
  3. Dataset pasang (PSMSL/BIG/IOC): sampling, gap, kualitas.
  4. Pipeline Bab 7: baseline persistence vs MLP vs LSTM/GRU; walk-forward.
  5. Evaluasi MAE/RMSE vs toleransi tinggi pasang; plot prediksi 1–7 hari; diskusi batas model.
- **Reproduksibilitas:** data + notebook (GitHub/Zenodo), **DOI bab**.
- **Latihan:** prediksi stasiun lain (mis. Semarang/rob) & bandingkan.
- **SEO:** "prediksi pasang surut LSTM", "pasang surut pontianak kapuas",
  "prediksi banjir rob machine learning".
- **Blog:** 3–4 artikel (konteks Kapuas; eksperimen; insight) + **video YouTube** (Fase II).

---

### Bab 9 — Studi Kasus: Prediksi Curah Hujan Stasiun BMKG
`book/manuscripts/ch-09-studi-kasus-curah-hujan-bmkg/master.md` · ±22–30 hal

- **Tujuan bab:** prediktor hujan terverifikasi metrik operasional; regresi & klasifikasi digabung.
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

- **Tujuan bab:** menjembatani model → operasional & menavigasi arah riset lanjut.
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

1. **Template seragam tiap bab:** Pembukaan masalah → Tujuan → Isi/konsep → Kode/notebook →
   Ringkasan kunci → Latihan → Referensi (IEEE, sertakan DOI) → Keyword SEO.
2. **Notasi & glosarium satu sumber:** istilah Indonesia + Inggris di pemunculan pertama
   (mis. "fungsi aktivasi (*activation function*)"); istilah sama di blog/buku/YouTube.
3. **Setiap bab berdiri sendiri** (sidebar "Prasyarat: Bab …"), tapi satu narasi & notasi.
4. **Reproduksibilitas:** seed tetap, versi TensorFlow, snapshot data di Zenodo,
   1 notebook per bab + template notebook kasus.
5. **Tone:** formal-hangat sesuai umbrella §3.1b; hindari hiperbola ("materi pengenalan",
   bukan klaim riset baru).
6. **Framing jujur di kasus:** DL dibandingkan dengan baseline (persistence/ARIMA/harmonik);
   hasil dilaporkan apa adanya.

---

## Catatan Revisi

- **Dari outline awal:** "Fungsi aktivasi" (bab mandiri) dibubarkan → muncul sebagai kebutuhan
  aplikasi di Bab 2–4; evaluasi domain ditambah Bab 5 agar kredibel di kalangan praktisi;
  data diperdalam Bab 6; kasus Kapuas & curah hujan jadi flagship setelah LSTM (Bab 8–9);
  operasional + arah riset digabung Bab 10.
- **Penyebab:** 10 iterasi review internal (audit struktur, learning-by-problem, framing tugas
  ML, evaluasi domain, adopsi pola code-first ala Bourke, data & reproduksibilitas, bobot &
  urutan kasus, sisi operasional, SEO & blog mapping, polish konsistensi).