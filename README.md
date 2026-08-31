# book/ — "Pengantar Deep Learning untuk Meteorologi"

Buku open-source berbahasa Indonesia. Setiap **bab = satu artikel** yang:
1. Ditulis sebagai Markdown (master) di `manuscripts/`.
2. Diberi **DOI Zenodo** (arsip versi PDF+DOCX di `releases/`).
3. Di-sinkronkan ke blog (`site/src/content/posts/`) agar versi "live" selalu aktual.

## Struktur

```
book/
├── manuscripts/
│   └── ch-01-pengantar-neuron/
│       ├── master.md        # MASTER: satu-satunya sumber kebenaran isi bab
│       └── refs.bib         # referensi IEEE (optional, dipakai saat PDF build)
├── releases/
│   └── v1.0.0/              # snapshot tiap rilis (PDF+DOCX) → dipakai untuk Zenodo
└── build/
    ├── generate.mjs         # generate PDF+DOCX per bab (butuh pandoc + LaTeX utk PDF)
    └── sync-to-blog.mjs     # sinkronkan master → blog Astro
```

## Frontmatter master.md

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

## Alur Kerja

1. Buat `manuscripts/ch-0N-<slug>/master.md`.
2. Message check: pastikan YAML frontmatter lengkap.
3. Generate rilis:
   ```
   node build/generate.mjs --version v1.0.0
   ```
4. Unggah `releases/v1.0.0/` ke Zenodo → copy DOI ke frontmatter.
5. Sinkronkan ke blog:
   ```
   node build/sync-to-blog.mjs
   ```
6. Set `status: published`, lalu build & deploy blog.

## Prasyarat

- **Pandoc** (wajib untuk PDF/DOCX): https://pandoc.org
- **LaTeX** (wajib hanya untuk output PDF; DOCX tidak butuh): mis. TinyTeX.

Tanpa kedua itu, script tetap bisa menyalin master & sync blog (tanpa output PDF/DOCX).