"""Publication-quality visualization and plotting suite for DCP/EDCP analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.info.mutual_information import compute_mi_profile, compute_mutual_information, information_loss_ratio
from src.utils.math_utils import wilson_score_interval

# Configure cohesive publication visual style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.edgecolor": "#cccccc",
    "grid.color": "#ebebeb",
    "grid.linestyle": "--",
})

PALETTE = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77", "#e7298a", "#66a61e"]


def plot_recovery_vs_truncation(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Plot recovery probability vs retained bits k for all moduli N.

    Args:
        df: Trial DataFrame or aggregated DataFrame.
        output_dir: Directory to save figure.

    Returns:
        Path to saved figure.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "recovery_vs_truncation.png"

    # Aggregate if raw trial df
    if "correct" in df.columns:
        grouped = df.groupby(["N", "k"]).agg(
            p_success=("correct", "mean"),
            success_count=("correct", "sum"),
            total=("correct", "count"),
        ).reset_index()
        grouped["ci_lower"] = grouped.apply(
            lambda r: wilson_score_interval(int(r["success_count"]), int(r["total"]))[0], axis=1
        )
        grouped["ci_upper"] = grouped.apply(
            lambda r: wilson_score_interval(int(r["success_count"]), int(r["total"]))[1], axis=1
        )
        # Mirror recovery (counts s OR N-s as correct)
        if "mirror_correct" in df.columns:
            mirror_grouped = df.groupby(["N", "k"]).agg(
                p_mirror=("mirror_correct", "mean"),
                mirror_count=("mirror_correct", "sum"),
                total=("mirror_correct", "count"),
            ).reset_index()
            mirror_grouped["mirror_ci_lower"] = mirror_grouped.apply(
                lambda r: wilson_score_interval(int(r["mirror_count"]), int(r["total"]))[0], axis=1
            )
            mirror_grouped["mirror_ci_upper"] = mirror_grouped.apply(
                lambda r: wilson_score_interval(int(r["mirror_count"]), int(r["total"]))[1], axis=1
            )
            grouped = grouped.merge(
                mirror_grouped[["N", "k", "p_mirror", "mirror_ci_lower", "mirror_ci_upper"]],
                on=["N", "k"], how="left"
            )
        else:
            grouped["p_mirror"] = grouped["p_success"]
            grouped["mirror_ci_lower"] = grouped["ci_lower"]
            grouped["mirror_ci_upper"] = grouped["ci_upper"]
    else:
        grouped = df.copy()
        if "p_success" not in grouped.columns and "recovery_prob" in grouped.columns:
            grouped["p_success"] = grouped["recovery_prob"]
        if "p_mirror" not in grouped.columns and "mirror_recovery_prob" in grouped.columns:
            grouped["p_mirror"] = grouped["mirror_recovery_prob"]
        elif "p_mirror" not in grouped.columns:
            grouped["p_mirror"] = grouped["p_success"]

    moduli = sorted(list(grouped["N"].unique()))

    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)

    for idx, N in enumerate(moduli):
        sub = grouped[grouped["N"] == N].sort_values("k")
        color = PALETTE[idx % len(PALETTE)]
        n_bits = max(1, (int(N) - 1).bit_length())

        # Exact recovery (solid line)
        ax.plot(
            sub["k"], sub["p_success"],
            "o-",
            label=f"N={N} exact",
            color=color,
            linewidth=2.2,
            markersize=7,
        )
        if "ci_lower" in sub.columns:
            ax.fill_between(sub["k"], sub["ci_lower"], sub["ci_upper"], color=color, alpha=0.12)

        # Mirror recovery — s OR N-s (dashed line, same color)
        if "p_mirror" in sub.columns:
            ax.plot(
                sub["k"], sub["p_mirror"],
                "s--",
                label=f"N={N} ±mirror",
                color=color,
                linewidth=1.8,
                markersize=6,
                alpha=0.75,
            )

        # Random baseline (dotted)
        ax.axhline(
            1.0 / N,
            color=color,
            linestyle=":",
            alpha=0.5,
            linewidth=1.1,
        )

    ax.set_xlabel("Retained Fourier Bits ($k$)")
    ax.set_ylabel("Recovery Probability $P(\\hat{s} = s)$")
    ax.set_title("DCP Secret Recovery vs. Fourier Truncation\n(solid = exact match, dashed = correct up to sign ambiguity)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(title="Modulus ($N$)", frameon=True, ncol=2, fontsize=9)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_mi_vs_truncation(
    output_dir: str | Path,
    moduli: Sequence[int] = (4, 8, 16, 32),
) -> Path:
    """Plot theoretical Mutual Information I(S; Y_k, B) vs retained bits k.

    Args:
        output_dir: Directory to save figure.
        moduli: Moduli to plot.

    Returns:
        Path to saved figure.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "mi_vs_truncation.png"

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    for idx, N in enumerate(moduli):
        prof = compute_mi_profile(N, mode="msb", include_flag=True)
        k_vals = sorted(prof.keys())
        mi_vals = [prof[k]["mi"] for k in k_vals]
        color = PALETTE[idx % len(PALETTE)]

        ax.plot(
            k_vals,
            mi_vals,
            "s-",
            label=f"N = {N} (max {np.log2(N):.1f} bits)",
            color=color,
            linewidth=2.2,
            markersize=7,
        )

    ax.set_xlabel("Retained Fourier Bits ($k$)")
    ax.set_ylabel("Mutual Information $I(S; Y_k, B)$ (bits)")
    ax.set_title("Exact Mutual Information vs. Fourier Truncation ($k$)")
    ax.legend(title="Modulus ($N$)", frameon=True)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_information_loss(
    output_dir: str | Path,
    moduli: Sequence[int] = (4, 8, 16, 32),
) -> Path:
    """Plot Information Loss Ratio (1 - I(S;Y_k)/I(S;Y)) vs retained bits k.

    Args:
        output_dir: Directory to save figure.
        moduli: Moduli to plot.

    Returns:
        Path to saved figure.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "information_loss_ratio.png"

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    for idx, N in enumerate(moduli):
        prof = compute_mi_profile(N, mode="msb", include_flag=True)
        k_vals = sorted(prof.keys())
        loss_vals = [prof[k]["info_loss_ratio"] for k in k_vals]
        color = PALETTE[idx % len(PALETTE)]

        ax.plot(
            k_vals,
            loss_vals,
            "^-",
            label=f"N = {N}",
            color=color,
            linewidth=2.2,
            markersize=7,
        )

    ax.set_xlabel("Retained Fourier Bits ($k$)")
    ax.set_ylabel("Information Loss Ratio $1 - I(S; Y_k) / I(S; Y)$")
    ax.set_title("Fourier Information Loss Ratio vs. Retained Bits ($k$)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(title="Modulus ($N$)", frameon=True)

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_bit_recovery_heatmap(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Generate per-bit recovery accuracy heatmaps for each modulus N.

    Args:
        df: Trial-level or aggregated DataFrame.
        output_dir: Target output directory.

    Returns:
        List of generated figure file paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    moduli = sorted(list(df["N"].unique()))

    for N in moduli:
        sub = df[df["N"] == N]
        bit_cols = [c for c in sub.columns if c.startswith("bit_correct_")]
        if not bit_cols:
            continue

        n_bits = len(bit_cols)
        k_vals = sorted(list(sub["k"].unique()))

        matrix = np.zeros((n_bits, len(k_vals)))
        for j, k_val in enumerate(k_vals):
            k_sub = sub[sub["k"] == k_val]
            for bit_i in range(n_bits):
                col = f"bit_correct_{bit_i}"
                if col in k_sub.columns:
                    matrix[bit_i, j] = float(k_sub[col].mean())

        fig, ax = plt.subplots(figsize=(6 + len(k_vals) * 0.5, 4.5), constrained_layout=True)
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            xticklabels=[f"k={k}" for k in k_vals],
            yticklabels=[f"Bit {i}" for i in range(n_bits)],
            cbar_kws={"label": "P(bit correct)"},
            ax=ax,
        )
        ax.set_title(f"Per-Bit Secret Recovery Accuracy ($N={N}$)")
        ax.set_xlabel("Fourier Truncation Level ($k$)")
        ax.set_ylabel("Bit Position (LSB = Bit 0)")

        file_path = out_dir / f"bit_recovery_heatmap_N{N}.png"
        fig.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(file_path)

    return saved_paths


def plot_posterior(
    posterior: Mapping[int, float],
    s_true: int,
    output_path: str | Path,
    title: str = "Posterior Distribution $P(s \\mid \\text{observations})$",
) -> Path:
    """Plot posterior probability distribution over candidate secrets in Z_N.

    Args:
        posterior: Dict mapping secret integer s to posterior probability.
        s_true: Ground truth secret.
        output_path: Save destination.
        title: Plot title.

    Returns:
        Path to saved figure.
    """
    save_p = Path(output_path)
    save_p.parent.mkdir(parents=True, exist_ok=True)

    secrets = sorted(posterior.keys())
    probs = [posterior[s] for s in secrets]
    s_hat = max(posterior, key=posterior.get)

    colors = []
    for s in secrets:
        if s == s_true and s == s_hat:
            colors.append("#2ca02c")  # Green: Correct & MAP
        elif s == s_true:
            colors.append("#1f77b4")  # Blue: True secret
        elif s == s_hat:
            colors.append("#d62728")  # Red: Incorrect MAP
        else:
            colors.append("#aec7e8")  # Light gray-blue

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bars = ax.bar(secrets, probs, color=colors, edgecolor="#333333", linewidth=0.8)

    ax.set_xlabel("Candidate Secret ($s \\in \\mathbb{Z}_N$)")
    ax.set_ylabel("Posterior Probability $P(s \\mid \\text{data})$")
    ax.set_title(title)
    ax.set_xticks(secrets)
    ax.set_ylim(0, max(probs) * 1.2 if probs else 1.0)

    # Custom legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#2ca02c", label=f"True Secret ($s={s_true}$, MAP)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#aec7e8", label="Other Candidates"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    fig.savefig(save_p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_p


def plot_summary_dashboard(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """Generate high-impact 4-panel publication dashboard combining all core metrics.

    Panels:
        A: Recovery Probability vs. k
        B: Exact Mutual Information I(S; Y_k) vs. k
        C: Information Loss Ratio vs. k
        D: Mean Bit Recovery Accuracy vs. k

    Args:
        df: Trial DataFrame or aggregated DataFrame.
        output_dir: Directory to save dashboard.

    Returns:
        Path to saved dashboard image.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / "dcp_core_dashboard.png"

    # Aggregated metrics by N and k
    grouped = df.groupby(["N", "k"]).agg(
        p_success=("correct", "mean"),
        bit_acc=("mean_bit_accuracy", "mean"),
        success_count=("correct", "sum"),
        total=("correct", "count"),
    ).reset_index()
    grouped["ci_lower"] = grouped.apply(
        lambda r: wilson_score_interval(int(r["success_count"]), int(r["total"]))[0], axis=1
    )
    grouped["ci_upper"] = grouped.apply(
        lambda r: wilson_score_interval(int(r["success_count"]), int(r["total"]))[1], axis=1
    )

    moduli = sorted(list(grouped["N"].unique()))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)

    # Panel A: Recovery Probability
    ax_a = axes[0, 0]
    for idx, N in enumerate(moduli):
        sub = grouped[grouped["N"] == N].sort_values("k")
        color = PALETTE[idx % len(PALETTE)]
        ax_a.plot(sub["k"], sub["p_success"], "o-", label=f"N={N}", color=color, linewidth=2)
        ax_a.fill_between(sub["k"], sub["ci_lower"], sub["ci_upper"], color=color, alpha=0.15)
        ax_a.axhline(1.0 / N, color=color, linestyle=":", alpha=0.5)
    ax_a.set_title("A: Secret Recovery Probability vs. $k$", fontweight="bold")
    ax_a.set_xlabel("Retained Bits ($k$)")
    ax_a.set_ylabel("$P(\\hat{s} = s)$")
    ax_a.legend()

    # Panel B: Mutual Information
    ax_b = axes[0, 1]
    for idx, N in enumerate(moduli):
        prof = compute_mi_profile(N, mode="msb", include_flag=True)
        k_vals = sorted(prof.keys())
        mi_vals = [prof[k]["mi"] for k in k_vals]
        color = PALETTE[idx % len(PALETTE)]
        ax_b.plot(k_vals, mi_vals, "s-", label=f"N={N}", color=color, linewidth=2)
    ax_b.set_title("B: Mutual Information $I(S; Y_k, B)$", fontweight="bold")
    ax_b.set_xlabel("Retained Bits ($k$)")
    ax_b.set_ylabel("Information (bits)")
    ax_b.legend()

    # Panel C: Information Loss Ratio
    ax_c = axes[1, 0]
    for idx, N in enumerate(moduli):
        prof = compute_mi_profile(N, mode="msb", include_flag=True)
        k_vals = sorted(prof.keys())
        loss_vals = [prof[k]["info_loss_ratio"] for k in k_vals]
        color = PALETTE[idx % len(PALETTE)]
        ax_c.plot(k_vals, loss_vals, "^-", label=f"N={N}", color=color, linewidth=2)
    ax_c.set_title("C: Information Loss Ratio $1 - I(S;Y_k)/I(S;Y)$", fontweight="bold")
    ax_c.set_xlabel("Retained Bits ($k$)")
    ax_c.set_ylabel("Loss Ratio")
    ax_c.set_ylim(-0.02, 1.02)
    ax_c.legend()

    # Panel D: Mean Bit Accuracy
    ax_d = axes[1, 1]
    for idx, N in enumerate(moduli):
        sub = grouped[grouped["N"] == N].sort_values("k")
        color = PALETTE[idx % len(PALETTE)]
        ax_d.plot(sub["k"], sub["bit_acc"], "d-", label=f"N={N}", color=color, linewidth=2)
    ax_d.axhline(0.5, color="gray", linestyle="--", label="Random Bit (0.5)")
    ax_d.set_title("D: Average Bit Recovery Accuracy", fontweight="bold")
    ax_d.set_xlabel("Retained Bits ($k$)")
    ax_d.set_ylabel("Mean Bit Accuracy")
    ax_d.set_ylim(0.45, 1.02)
    ax_d.legend()

    fig.suptitle("DCP Fourier Information Truncation & Recovery Analysis", fontsize=15, fontweight="bold")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path
