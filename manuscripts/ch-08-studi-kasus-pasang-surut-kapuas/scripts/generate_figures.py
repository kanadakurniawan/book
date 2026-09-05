#!/usr/bin/env python3
"""
Hasilkan Gambar 8.2 (prediksi vs aktual 7 hari) dan Gambar 8.3 (residu per fase
pasang) untuk Bab 8, dari data sample Cilacap.

CATATAN KEJUJURAN:
  - Sumber data: data/sample/cili_1y_hourly.csv (sintetik deterministik,
    komponen harmonik mirip Cilacap). Bukan observasi IOC/UHSLC.
  - "Prediksi LSTM" di sini adalah hasil dari model MLP kecil yang dilatih di
    skrip ini (TensorFlow dibutuhkan). Jika TF tidak tersedia, skrip fallback
    membuat prediksi berbasis persistence (naive) sebagai ilustrasi visual.
  - Gambar yang dihasilkan dipakai untuk pembelajaran membaca plot, bukan untuk
    klaim performa model di data nyata.

Penggunaan:
  python generate_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)

HERE = Path(__file__).resolve().parent.parent
DATA = HERE / "data" / "sample" / "cili_1y_hourly.csv"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def load_series() -> pd.Series:
    df = pd.read_csv(DATA, parse_dates=["time"]).set_index("time")
    s = df["tinggi"].interpolate(limit=6).dropna()
    return s


def predict_with_tf(ser: np.ndarray, w: int = 168, h: int = 1) -> np.ndarray:
    """Latih MLP kecil dan prediksi pada test split. Mengembalikan array prediksi
    selaras dengan index window test."""
    import tensorflow as tf  # noqa: WPS433
    tf.random.set_seed(42)

    def make_xy(arr, w, h):
        X, y = [], []
        for i in range(len(arr) - w - h + 1):
            X.append(arr[i : i + w])
            y.append(arr[i + w + h - 1])
        return np.array(X), np.array(y)

    X, y = make_xy(ser, w, h)
    n = len(X)
    ntr = int(n * 0.8)
    Xtr, ytr = X[:ntr], y[:ntr]
    Xte, yte = X[ntr:], y[ntr:]
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation="relu", input_shape=(w,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(Xtr, ytr, validation_split=0.15, epochs=15, batch_size=32, verbose=0)
    p = model.predict(Xte, verbose=0).ravel()
    return p, yte, Xte


def predict_persistence(ser: np.ndarray) -> np.ndarray:
    """Fallback: prediksi = persistence (geser 24 jam ke depan, disetahunkan).
    Untuk deret periodik, ini sebenarnya baseline yang kuat."""
    return ser[24:]


def plot_figure_8_2(ser: pd.Series, pred: np.ndarray, src: str) -> Path:
    """Plot prediksi vs aktual untuk 7 hari terakhir."""
    hari = 7
    sample_per_hour = 1
    n = hari * 24 * sample_per_hour
    aktual = ser.values[-n:]
    # Ambil prediksi sepanjang mungkin dari test split; potong n terakhir.
    pred_window = pred[-n:] if len(pred) >= n else pred
    if len(pred_window) < n:
        pred_window = np.concatenate([
            np.full(n - len(pred_window), aktual[0]),
            pred_window,
        ])
    t = ser.index[-n:]

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(t, aktual, label="aktual", lw=1.4, color="#1f4e79")
    ax.plot(t, pred_window, label=f"prediksi ({src})", lw=1.1, ls="--",
            alpha=0.85, color="#e0893d")
    ax.set_xlabel("Waktu (UTC)")
    ax.set_ylabel("Tinggi muka air (m, relatif)")
    ax.set_title(f"Prediksi vs aktual — {hari} hari terakhir (Cilacap, data sample)")
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()

    out_png = FIG / "fig-8-2-forecast-7hari.png"
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    # opsional webp kecil
    try:
        out_webp = FIG / "fig-8-2-forecast-7hari.webp"
        fig2, ax2 = plt.subplots(figsize=(10, 3.6))
        ax2.plot(t, aktual, label="aktual", lw=1.4, color="#1f4e79")
        ax2.plot(t, pred_window, label=f"prediksi ({src})", lw=1.1, ls="--",
                 alpha=0.85, color="#e0893d")
        ax2.set_xlabel("Waktu (UTC)")
        ax2.set_ylabel("Tinggi muka air (m)")
        ax2.set_title(f"Prediksi vs aktual — {hari} hari terakhir (Cilacap, data sample)")
        ax2.legend(loc="upper right")
        ax2.grid(alpha=0.25)
        fig2.autofmt_xdate()
        fig2.tight_layout()
        fig2.savefig(out_webp, dpi=130, format="webp")
        plt.close(fig2)
    except Exception:
        pass
    return out_png


def plot_figure_8_3(ser: pd.Series, pred: np.ndarray, src: str) -> Path:
    """Plot residu per fase pasang (scatter error vs tinggi aktual)."""
    hari = 7
    n = hari * 24
    aktual = ser.values[-n:]
    pred_window = pred[-n:] if len(pred) >= n else pred
    if len(pred_window) < n:
        pred_window = np.concatenate([
            np.full(n - len(pred_window), aktual[0]),
            pred_window,
        ])
    residu = pred_window - aktual

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

    # Panel kiri: scatter error vs tinggi aktual
    ax = axes[0]
    ax.scatter(aktual, residu, s=10, alpha=0.55, color="#4a90e2")
    ax.axhline(0, color="#333", lw=0.8, ls="--")
    ax.set_xlabel("Tinggi aktual (m)")
    ax.set_ylabel("Residu (prediksi − aktual), m")
    ax.set_title("Residu vs amplitudo")
    ax.grid(alpha=0.25)

    # Panel kanan: residu terhadap fase pasang (indeks jam dalam siklus ~12,42 jam)
    # fase = jam dalam siklus M2 (12,42 jam), dilipat ke [0, 1)
    fase = (np.arange(n) % int(12.42)) / 12.42
    ax = axes[1]
    ax.scatter(fase, residu, s=10, alpha=0.55, color="#27ae60")
    ax.axhline(0, color="#333", lw=0.8, ls="--")
    ax.set_xlabel("Fase dalam siklus M2 (0 = awal siklus)")
    ax.set_ylabel("Residu (prediksi − aktual), m")
    ax.set_title("Residu vs fase pasang")
    ax.grid(alpha=0.25)

    fig.suptitle(f"Residu {hari} hari terakhir — Cilacap, data sample, {src}",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    out_png = FIG / "fig-8-3-residu.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    try:
        out_webp = FIG / "fig-8-3-residu.webp"
        fig.savefig(out_webp, dpi=130, format="webp", bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass
    return out_png


def main() -> int:
    if not DATA.exists():
        raise SystemExit(f"Sample data tidak ditemukan: {DATA}. Jalankan generate_sample.py dulu.")
    ser = load_series()
    arr = ser.values

    src = "MLP"
    try:
        pred, yte, _Xte = predict_with_tf(arr, w=168, h=1)
        # selaraskan panjang dengan test window; kita pakai pred sepanjang mungkin
        # (skala window test ~20% data). Untuk plot 7 hari terakhir, ambil tail.
        pred = pred  # biarkan apa adanya; plot akan pakai tail pred
    except Exception as e:
        print(f"[info] TensorFlow tidak tersedia ({e.__class__.__name__}); pakai persistence.")
        src = "persistence (fallback)"
        pred = predict_persistence(arr)

    out2 = plot_figure_8_2(ser, pred, src)
    out3 = plot_figure_8_3(ser, pred, src)
    print(f"Tulis {out2}")
    print(f"Tulis {out3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
