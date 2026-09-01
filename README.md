# Pengantar Deep Learning untuk Meteorologi

> **"Deep learning untuk meteorologi, dari praktisi untuk praktisi di Indonesia."**

Buku pengenalan open-source berbahasa Indonesia yang ditulis oleh **Kanada Kurniawan** —
Meteorology Officer di BMKG (Stasiun Meteorologi Kelas III Maritim Dwikora, Pontianak) dengan
latar riset pengembangan model neural network untuk prediksi cuaca, pasang surut, dan
hidrologi.

Ini adalah materi **pengenalan**, bukan hasil riset baru. Gagasannya sederhana: banyak
praktisi dan mahasiswa kebumian ingin belajar deep learning, tetapi materi yang beredar
terlalu berorientasi ilmu komputer dan jauh dari konteks data Indonesia. Buku ini hadir
untuk mengisi celah itu — dengan contoh data lokal dan bahasa yang bisa dipahami.

## Konsep

Setiap **bab adalah satu artikel** yang:

1. Ditulis sebagai **Markdown (master)** di `manuscripts/` — satu-satunya sumber kebenaran isi.
2. Diberi **DOI Zenodo** per rilis (PDF+DOCX diarsipkan di `releases/`), sehingga setiap bab
   dapat dikutip dengan stabil seperti publikasi.
3. Di-sinkronkan ke blog agar versi "live" selalu aktual:
   ```
   node build/sync-to-blog.mjs
   ```

> Rencana detil isi buku (isi per bab, notebook, latihan, SEO, blog mapping, pacing rilis)
> ada di **`outline.md`** — sumber kebenaran perencanaan, dipakai saat menulis tiap bab.

Pendekatan "buku yang hidup" ini memungkinkan materi terus diperbarui tanpa memutus tautan
sitasi — versi baru naik di Zenodo, concept DOI tetap sama.

## Struktur Repositori

```
book/
├── outline.md               # OUTLINE: rencana & detil isi 10 bab (sumber perencanaan)
├── front-matter/
│   └── cara-memakai-buku.md  # FRONT MATTER: panduan baca (tidak dirilis ke Zenodo)
├── manuscripts/
│   └── ch-01-pengantar-deep-learning-meteorologi/
│       ├── master.md        # MASTER: sumber kebenaran isi bab
│       └── refs.bib         # referensi (dipakai saat PDF build via citeproc)
├── releases/
│   └── v1.0.0/              # snapshot tiap rilis (PDF+DOCX) → untuk Zenodo
└── build/
    ├── generate.mjs         # master.md → PDF + DOCX per versi
    └── sync-to-blog.mjs     # sinkronkan master → blog (site/src/content/posts)
```

## Frontmatter `master.md`

| Field | Kegunaan |
|---|---|
| `title` | Judul bab (blog + PDF) |
| `description` | Ringkasan di blog & metadata PDF |
| `pubDate` | Tanggal rilis (format ISO `YYYY-MM-DD`) |
| `categories`, `tags` | Kategori/SEO blog |
| `version` | Versi bab (semver) |
| `doi` | DOI Zenodo bab (placeholder sampai dirilis) |
| `status` | `draft` (belum live) atau `published` |
| `chapter` | Nomor bab |

## Alur Kerja Rilis Satu Bab

1. Tulis `manuscripts/ch-0N-<slug>/master.md` lengkap dengan frontmatter.
2. Review isi dengan teliti (beberapa putaran) sebelum dirilis.
3. Generate rilis:
   ```
   node build/generate.mjs --version v1.0.0
   ```
4. Unggah `releases/v1.0.0/` ke Zenodo → salin DOI ke frontmatter.
5. Sinkronkan ke blog:
   ```
   node build/sync-to-blog.mjs
   ```
6. Set `status: published`, lalu build & deploy blog.

## Prasyarat

- **Pandoc** (wajib untuk output PDF/DOCX): https://pandoc.org
- **LaTeX** (hanya untuk output PDF; DOCX tidak butuh): mis. TinyTeX.

Tanpa keduanya, script tetap bisa sinkronkan master ke blog (tanpa hasil PDF/DOCX).

## Terhubung dengan Brand

- Blog & versi live bab: **kanadakurniawan.com**
- Situs web (deploy): repo `kanadakurniawan/site`

## Lisensi

Dokumen buku © Kanada Kurniawan. Silakan disebarluaskan, dikutip, dan dimanfaatkan untuk
belajar — upayakan tetap memberi kredit melalui DOI bab masing-masing.