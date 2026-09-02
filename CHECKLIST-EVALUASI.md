# Checklist Evaluasi Internal — Buku *Pengantar Deep Learning untuk Meteorologi*

> Wajib lolos **seluruh bab** sebelum publikasi: DOI Zenodo, ISBN, GitHub publik, dan blog.
> Referensi silang: perincian di `outline.md` → "Alur Kerja Penulisan & Evaluasi Internal".
> Skala: ✅ lolos · ⚠️ perlu perbaikan · ❌ gagal. Tulis catatan di kolom lembar kerja.

## A. Isi & Keilmuan

- [ ] Semua klaim teknis disitasi; tidak ada pernyataan substantif tanpa sumber.
- [ ] Ketepatan istilah: istilah Indonesia + Inggris benar, seragam di seluruh buku.
- [ ] Anti-overhype: semua hasil DL dibandingkan baseline; disclaimer "materi pengenalan" ada.
- [ ] Framing jujur di kasus: ML untuk prakiraan cepat/gap, bukan klaim riset baru.
- [ ] Angka/fakta bisa dilacak ke sumber (bukan hafalan/tebakan).

## B. Struktur & Konsistensi

- [ ] Template seragam tiap bab: Pembukaan masalah → Tujuan → Isi/konsep → Kode/notebook →
      Ringkasan kunci → Latihan → Referensi → Keyword SEO.
- [ ] **Tujuan Pembelajaran** (3–5 butir aksi) ada di awal bab dan konsisten dengan latihan
      (constructive alignment).
- [ ] Sidebar "Prasyarat: Bab …" benar untuk tiap bab.
- [ ] Notasi & glosarium satu sumber (tidak ada istilah ganda).
- [ ] Kata isi sesuai target volume (3.000–4.500/bab); tidak ada bab terlalu kurus/gendut.
- [ ] Math diketik konsisten; code block sesuai style.

## C. Sitasi

- [ ] `[n]` di teks ≡ daftar References ≡ `refs.bib` (urut incremental pertama-muncul).
- [ ] DOI valid (uji via `doi.org/<doi>`); ISBN/arXiv tercantum bila tidak ada DOI.
- [ ] URL punya tanggal akses.
- [ ] Tidak ada self-citation berlebihan.

## D. Kode & Reproduksibilitas

- [ ] Semua notebook dapat dijalankan dari awal–akhir (Colab) tanpa error.
- [ ] Seed tetap, versi TensorFlow tercatat.
- [ ] Data kasus tersedia / telah diunggah (dengan lisensi).

## E. Build & Output

- [ ] `node build/generate.mjs` menghasilkan PDF+DOCX tanpa error (butuh Pandoc+LaTeX
      pada mesin rilis).
- [ ] Total halaman realistis (±220 penarget); front/back matter siap.
- [ ] `bookDOI` terisi di semua `master.md` setelah DOI dibuat.

---

## Ringkasan Status per Bab

| Bab | A | B | C | D | E | Siap publikasi? |
|-----|---|---|---|---|---|-----------------|
| 1 |   |   |   |   |   |                 |
| 2 |   |   |   |   |   |                 |
| 3 |   |   |   |   |   |                 |
| 4 |   |   |   |   |   |                 |
| 5 |   |   |   |   |   |                 |
| 6 |   |   |   |   |   |                 |
| 7 |   |   |   |   |   |                 |
| 8 |   |   |   |   |   |                 |
| 9 |   |   |   |   |   |                 |
| 10 |   |   |   |   |   |                 |

> Rule: seluruh bab harus "✅" di semua bagian sebelum lanjut ke Fase 3 (penerbitan).