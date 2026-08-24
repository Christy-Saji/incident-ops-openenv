"""Reward curve and component plotting utilities."""

from __future__ import annotations

import os


def plot_reward_curve(
    log_path: str,
    out_path: str,
    smooth_window: int = 10,
) -> None:
    """Two-panel training diagnostic.

    Top:    reward — raw (faint), smoothed (bold), linear trend (dashed).
    Bottom: reward_std — the mode-collapse alarm.

    Read the bottom panel first. GRPO's advantage is (r - group_mean) / group_std,
    so reward_std ~ 0 means every completion in the group was identical, every
    advantage was 0, and no gradient flowed regardless of what the top panel
    shows. A flat top panel with a healthy reward_std is a reward-design problem;
    a flat top panel with reward_std at 0 is mode collapse.
    """
    import matplotlib
    import numpy as np
    import pandas as pd
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
    ALARM_RED  = "#C44E52"

    has_std = "reward_std" in df.columns and df["reward_std"].notna().any()

    fig, axes = plt.subplots(
        2 if has_std else 1, 1,
        figsize=(12, 7.5 if has_std else 5),
        facecolor="white",
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]} if has_std else None,
    )
    axes = axes if has_std else [axes]
    ax = axes[0]

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
    ax.set_ylabel("Reward", fontsize=9)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.8)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

    if has_std:
        ax_std = axes[1]
        std_vals = df["reward_std"].values
        ax_std.plot(steps, std_vals, color=ALARM_RED, linewidth=1.6,
                    label="reward_std (within-group)")
        ax_std.axhline(0, color=ALARM_RED, linewidth=1.0, linestyle="--", alpha=0.8)
        ax_std.fill_between(steps, 0, std_vals, color=ALARM_RED, alpha=0.15)

        ax_std.annotate(
            "mode collapse — no gradient",
            xy=(steps[0] if len(steps) else 0, 0),
            xytext=(4, 6), textcoords="offset points",
            fontsize=8, color=ALARM_RED, fontweight="bold", va="bottom",
        )
        ax_std.set_title(
            "Within-Group Reward Std — advantage is (r - mean) / std, "
            "so std ~ 0 means zero gradient",
            fontsize=9, pad=6,
        )
        ax_std.set_ylabel("reward_std", fontsize=9)
        ax_std.legend(fontsize=8, loc="upper right", framealpha=0.8)
        ax_std.grid(True, alpha=0.3)
        ax_std.tick_params(labelsize=8)
        ax_std.set_ylim(bottom=0)

    axes[-1].set_xlabel("Training Step", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Reward curve saved to {out_path}")


def plot_reward_components(
    log_path: str,
    out_path: str,
) -> None:
    """Bar chart of mean reward per reward function across all training steps."""
    import matplotlib
    import pandas as pd
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
    ax.barh(labels, means.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Mean Reward Contribution", fontsize=9)
    ax.set_title("Reward Component Contributions (mean across all steps)", fontsize=10, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Component chart saved to {out_path}")
