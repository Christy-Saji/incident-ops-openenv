"""Reward curve and component plotting utilities."""

from __future__ import annotations

import os


def plot_reward_curve(
    log_path: str,
    out_path: str,
    smooth_window: int = 10,
) -> None:
    """Single-panel reward curve: raw (faint), smoothed (bold), linear trend (dashed)."""
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists(log_path):
        print(f"[plot] Reward log not found: {log_path}")
        return

    df = pd.read_csv(log_path)
    for col in df.columns:
        if col != "step":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["step", "reward"]).sort_values("step").reset_index(drop=True)
    steps = df["step"].values
    raw   = df["reward"].values

    smoothed = pd.Series(raw).rolling(smooth_window, min_periods=1, center=True).mean().values

    mask = ~np.isnan(raw)
    trend = None
    if mask.sum() >= 2:
        sl, ic = np.polyfit(steps[mask], raw[mask], 1)
        trend = sl * steps + ic

    BASE_BLUE  = "#4C72B0"
    LIGHT_BLUE = "#A8C4E0"

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    fig.suptitle(
        "GRPO Training — DevOps Incident Triage SRE Agent\n"
        "SFT warm-start + GRPO (9 reward signals)",
        fontsize=11, fontweight="bold",
    )

    ax.plot(steps, raw,      color=LIGHT_BLUE, alpha=0.35, linewidth=0.8, label="raw")
    ax.plot(steps, smoothed, color=BASE_BLUE,  linewidth=2.2,
            label=f"smoothed (w={smooth_window})")
    if trend is not None:
        ax.plot(steps, trend, color="#888888", linewidth=1.2,
                linestyle="--", alpha=0.7, label="trend")

    ax.axhline(0, color="#cccccc", linewidth=0.6, linestyle=":")
    ax.set_title("Overall Reward", fontsize=10, pad=6)
    ax.set_xlabel("Training Step", fontsize=9)
    ax.set_ylabel("Reward", fontsize=9)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Reward curve saved to {out_path}")


def plot_reward_components(
    log_path: str,
    out_path: str,
) -> None:
    """Bar chart of mean reward per reward function across all training steps."""
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists(log_path):
        print(f"[plot] Reward log not found: {log_path}")
        return

    df = pd.read_csv(log_path)
    component_cols = [c for c in df.columns if c.startswith("reward_") and c != "reward"]
    for col in component_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    means = df[component_cols].mean().sort_values(ascending=True)
    colors = ["#d73027" if v < 0 else "#4575b4" for v in means]

    # Clean up label names for display
    labels = [
        col.replace("reward_", "").replace("_func", "").replace("_", " ")
        for col in means.index
    ]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    bars = ax.barh(labels, means.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Mean Reward Contribution", fontsize=9)
    ax.set_title("Reward Component Contributions (mean across all steps)", fontsize=10, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Component chart saved to {out_path}")
