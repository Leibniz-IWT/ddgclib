from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
ROWS_PATH = ROOT / "out/sphere_fheron_case11_github_twophase_eos_preview/compressible_rows.json"
OUT_DIR = ROOT / "out/github_case_a2_vs_t"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = json.loads(ROWS_PATH.read_text())
    t = np.array([row["t"] for row in rows], dtype=float)
    a_sim = np.array([row["shape_amplitude_fit"] for row in rows], dtype=float)
    a_theory = np.array([row["shape_amplitude_theory"] for row in rows], dtype=float)

    rmse = float(np.sqrt(np.mean((a_sim - a_theory) ** 2)))
    maxerr = float(np.max(np.abs(a_sim - a_theory)))

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.plot(t, a_theory, "k--", lw=2.0, label="Rayleigh theory")
    ax.plot(t, a_sim, color="#1f77b4", lw=2.0, label="ddgclib GitHub case: fitted $a_2$")
    ax.scatter([t[-1]], [a_sim[-1]], color="#d62728", s=40, zorder=5)
    ax.set_ylabel("$a_2$")
    ax.set_title("ddgclib GitHub oscillating-droplet case: $a_2(t)$ vs Rayleigh theory")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.02,
        0.04,
        (
            f"RMSE = {rmse:.3f}\n"
            f"max |error| = {maxerr:.3f}\n"
            f"final sim = {a_sim[-1]:.3f}\n"
            f"final theory = {a_theory[-1]:.3f}"
        ),
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
    )

    axz = axes[1]
    axz.plot(t, a_theory, "k--", lw=2.0, label="Rayleigh theory")
    axz.plot(t, a_sim, color="#1f77b4", lw=2.0, label="GitHub fitted $a_2$")
    axz.axhline(0.0, color="0.7", lw=0.8)
    axz.set_ylim(-0.08, 0.08)
    axz.set_xlabel("t [s]")
    axz.set_ylabel("$a_2$ zoom")
    axz.set_title("Zoom to Rayleigh amplitude scale")
    axz.grid(True, alpha=0.25)
    axz.legend(frameon=False, loc="upper right")

    png = OUT_DIR / "github_case_a2_vs_t.png"
    fig.savefig(png, dpi=180)
    plt.close(fig)
    print(png)
    print(
        f"rows={len(rows)} t_end={t[-1]:.6f} "
        f"sim_final={a_sim[-1]:.6f} theory_final={a_theory[-1]:.6f} "
        f"rmse={rmse:.6f} maxerr={maxerr:.6f}"
    )


if __name__ == "__main__":
    main()
