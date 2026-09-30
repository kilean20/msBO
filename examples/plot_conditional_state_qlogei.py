"""Create the CS-qLogEI method diagram and matched-benchmark figure."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


HERE = Path(__file__).resolve().parent
BENCHMARK = HERE / "benchmark" / "switch_acquisition_modes_20seeds_noiseaware.csv"


def add_box(ax, xy, width, height, text, color):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        facecolor=color,
        edgecolor="#243247",
        linewidth=1.3,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=10,
        color="#152033",
        linespacing=1.25,
    )
    return box


def flow_figure():
    fig, ax = plt.subplots(figsize=(12, 3.7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    boxes = [
        add_box(ax, (0.2, 1.35), 1.75, 1.3,
                "Candidate controls\n$X=(x_1,\ldots,x_q)$", "#d8ebff"),
        add_box(ax, (2.45, 1.35), 2.05, 1.3,
                "Scheduled state\n$s=$ `nominal`\n(categorical label)", "#fff0c7"),
        add_box(ax, (5.0, 1.35), 2.05, 1.3,
                "Sample possible noisy\nreadings $y_s(X)$", "#e4f5df"),
        add_box(ax, (7.55, 1.35), 2.0, 1.3,
                "Gaussian conditional\nmean of all responses", "#eee4ff"),
        add_box(ax, (10.05, 1.35), 1.75, 1.3,
                "Composite objective\n+ stable qLogEI", "#ffdede"),
    ]
    for left, right in zip(boxes[:-1], boxes[1:]):
        y = left.get_y() + left.get_height() / 2
        arrow = FancyArrowPatch(
            (left.get_x() + left.get_width() + 0.05, y),
            (right.get_x() - 0.05, y),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color="#40526d",
        )
        ax.add_patch(arrow)

    ax.text(
        6,
        3.45,
        "Conditional-state qLogEI: value only what the scheduled state can observe",
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
        color="#152033",
    )
    ax.text(
        7.55,
        0.55,
        "Cross-state GP covariance transfers information; the state identifier itself is never interpolated.",
        ha="center",
        va="center",
        fontsize=9.5,
        color="#40526d",
    )
    path = HERE / "conditional_state_qlogei_flow.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def benchmark_figure():
    frame = pd.read_csv(BENCHMARK)
    modes = ["global", "conditional"]
    colors = {"global": "#4c78a8", "conditional": "#e45756"}
    paired = frame.pivot(index="seed", columns="mode", values="true_objective")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    ax = axes[0]
    loss = -paired[modes]
    for _, row in loss.iterrows():
        ax.plot([0, 1], row.values, color="#a9b2c1", alpha=0.55, linewidth=1)
        ax.scatter([0, 1], row.values, color=[colors[m] for m in modes], s=18, zorder=3)
    medians = loss.median()
    ax.scatter([0, 1], medians.values, marker="D", s=75, color=[colors[m] for m in modes],
               edgecolor="black", linewidth=0.7, zorder=4, label="median")
    ax.set_yscale("log")
    ax.set_xticks([0, 1], ["Global", "Conditional"])
    ax.set_ylabel("Final true loss (lower is better, log scale)")
    ax.set_title("20 matched seeds: 10–10 wins")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    data = [frame.loc[frame["mode"] == mode, "query_total_sec"] for mode in modes]
    bp = ax.boxplot(data, patch_artist=True, labels=["Global", "Conditional"], showfliers=True)
    for patch, mode in zip(bp["boxes"], modes):
        patch.set_facecolor(colors[mode])
        patch.set_alpha(0.72)
    ax.set_ylabel("Total query time per 4-state cycle (s)")
    ax.set_title("Median: 1.29 s vs 2.20 s")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[2]
    x = np.arange(2)
    width = 0.34
    bridge = frame.groupby("mode")["bridge_compute_max_sec"].median().reindex(modes)
    prefetch = frame.groupby("mode")["overlap_compute_max_sec"].median().reindex(modes)
    ax.bar(x - width / 2, bridge, width, label="bridge query", color="#72b7b2")
    ax.bar(x + width / 2, prefetch, width, label="prefetch query", color="#f2cf5b")
    ax.axhline(4.0, color="#333333", linestyle="--", linewidth=1.3,
               label="1 s ramp + 3 s read")
    ax.set_xticks(x, ["Global", "Conditional"])
    ax.set_ylabel("Median of per-run maximum (s)")
    ax.set_ylim(0, 4.5)
    ax.set_title("Computation fits the machine window")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Switch-scheduler acquisition benchmark (2 controls, 4 states, 2 diagnostics/state)",
                 fontsize=14, weight="bold")
    fig.tight_layout()
    path = HERE / "benchmark" / "conditional_state_qlogei_benchmark.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    for output in (flow_figure(), benchmark_figure()):
        print(output)
