"""Puts two trained networks side by side on the same long run.

The simulation is run for 1200 snapshots and each network is left to predict
all of them on its own from the first one. The result is an animation with
the simulation and both networks, pictures at a few moments, and a chart of
how much the flow behind the obstacle is moving in each.

Run from the project folder:
    python scripts/compare_versions.py runs/model_v1.pt runs/model.pt [reynolds]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.animation import FuncAnimation, PillowWriter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import AQUA, BLUE, CENTRE_X, MUTED, ORANGE, RESULTS, SURFACE, predict, spin, style
from flowsim.lbm import Flow, cylinder
from flowsim.model import FlowNet

FRAMES = 1200
NAMES = ("Simulation", "Version 1", "Version 2")

if __name__ == "__main__":
    reynolds = float(sys.argv[3]) if len(sys.argv) > 3 else 160.0

    # The same cylinder as in the Reynolds sweep of the test set.
    mask = cylinder(768, 256, 170, 128, 18)
    flow = Flow(mask, reynolds, 36.0)
    flow.step(40000)
    frames = []
    for _ in range(FRAMES + 1):
        flow.step(100)
        frames.append(F.avg_pool2d(flow.velocity() / flow.speed, 2)[0])
    true = torch.stack(frames)
    case = dict(velocity=true, solid=F.avg_pool2d(mask.float()[None, None], 2)[0, 0], reynolds=reynolds)

    runs = [true[1:]]
    for path in sys.argv[1:3]:
        model = FlowNet().cuda()
        model.load_state_dict(torch.load(ROOT / path))
        model.eval()
        runs.append(predict(model, case, FRAMES))
    solid = case["solid"]
    RESULTS.mkdir(exist_ok=True)

    # How much sideways motion there is behind the obstacle, averaged over windows of 25 snapshots.
    wake = slice(CENTRE_X + 20, CENTRE_X + 200)
    activity = lambda v: v[:, 1, wake].abs().mean(dim=(1, 2)).unfold(0, 25, 25).mean(dim=1).cpu().numpy()
    moments = 25 * np.arange(1, FRAMES // 25 + 1)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160, facecolor=SURFACE)
    for run, name, colour, dashes in zip(runs, NAMES, (BLUE, ORANGE, AQUA), ("solid", (0, (6, 2)), (0, (1, 1.5)))):
        ax.plot(moments, activity(run), color=colour, linewidth=2, linestyle=dashes, label=name)
    style(ax, "snapshots predicted in a row by each network on its own", "sideways motion behind the obstacle\n(fraction of the inflow speed)")
    ax.set_xlim(0, FRAMES)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=10, labelcolor=MUTED, loc="lower left")
    fig.savefig(RESULTS / "versions_activity.png", bbox_inches="tight")
    plt.close(fig)

    limit = 0.8 * np.nanmax(np.abs(spin(true[-1], solid)[:, CENTRE_X + 12 :]))
    cmap = plt.get_cmap("berlin").copy()
    cmap.set_bad("#8a8f98")

    # Pictures of the three at a few moments.
    shown = (100, 300, 500, 800, 1200)
    fig, axes = plt.subplots(3, len(shown), figsize=(16, 3.8), dpi=160, facecolor="#101014")
    fig.subplots_adjust(left=0.07, right=0.998, top=0.92, bottom=0.01, hspace=0.05, wspace=0.02)
    for column, moment in enumerate(shown):
        for row, run in enumerate(runs):
            ax = axes[row, column]
            ax.imshow(spin(run[moment - 1], solid), cmap=cmap, vmin=-limit, vmax=limit, origin="lower", interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(f"snapshot {moment}", color="#c3c2b7", fontsize=10)
            if column == 0:
                ax.set_ylabel(NAMES[row], color="#f0efec", fontsize=10, rotation=0, ha="right", va="center", labelpad=8)
    fig.savefig(RESULTS / "versions_snapshots.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    # The three in motion.
    fig, axes = plt.subplots(3, 1, figsize=(7.68, 8.0), dpi=75, facecolor="#101014")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.01, hspace=0.13)
    images = []
    for ax, run, name in zip(axes, runs, NAMES):
        ax.axis("off")
        ax.set_title(name, color="#f0efec", fontsize=11, loc="left", pad=4)
        images.append(ax.imshow(spin(run[0], solid), cmap=cmap, vmin=-limit, vmax=limit, origin="lower", interpolation="bilinear"))
    fig.suptitle(f"Reynolds {reynolds:.0f}, each network predicting on its own", color="#c3c2b7", fontsize=10, x=0.99, ha="right")
    counter = fig.text(0.01, 0.972, "", color="#c3c2b7", fontsize=10)

    def update(k):
        for image, run in zip(images, runs):
            image.set_data(spin(run[k], solid))
        counter.set_text(f"snapshot {k + 1}")

    FuncAnimation(fig, update, frames=range(0, FRAMES, 8)).save(RESULTS / "versions.gif", writer=PillowWriter(fps=12))
    plt.close(fig)
    print("activity at the end:", {name: round(float(activity(run)[-1]), 3) for name, run in zip(NAMES, runs)})
