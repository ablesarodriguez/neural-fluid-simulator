"""Checks how long the network keeps going: a run far longer than its training examples.

The simulation is run for 1500 snapshots (the training examples have 600) and
the network is left to predict all of them on its own from the first one. The
result is an animation of both, pictures at a few moments, and a chart of how
much the flow behind the obstacle is moving in each.

Run from the project folder:  python scripts/long_run.py [reynolds]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate import BLUE, CENTRE_X, ORANGE, RESULTS, MUTED, SURFACE, predict, save_animation, spin, style
from flowsim.lbm import Flow, cylinder
from flowsim.model import FlowNet

REYNOLDS = float(sys.argv[1]) if len(sys.argv) > 1 else 160.0
FRAMES = 1500

if __name__ == "__main__":
    # The same cylinder as in the Reynolds sweep of the test set, simulated for much longer.
    mask = cylinder(768, 256, 170, 128, 18)
    flow = Flow(mask, REYNOLDS, 36.0)
    flow.step(20000)
    frames = []
    for _ in range(FRAMES + 1):
        flow.step(100)
        frames.append(F.avg_pool2d(flow.velocity() / flow.speed, 2)[0])
    true = torch.stack(frames)
    case = dict(velocity=true, solid=F.avg_pool2d(mask.float()[None, None], 2)[0, 0], reynolds=REYNOLDS)

    model = FlowNet().cuda()
    model.load_state_dict(torch.load(ROOT / "runs" / "model.pt"))
    model.eval()
    predicted = predict(model, case, FRAMES)
    true = true[1:]

    # How much sideways motion there is behind the obstacle, averaged over windows of 25 snapshots.
    wake = slice(CENTRE_X + 20, CENTRE_X + 200)
    activity = lambda v: v[:, 1, wake].abs().mean(dim=(1, 2)).unfold(0, 25, 25).mean(dim=1).cpu().numpy()
    moments = 25 * np.arange(1, FRAMES // 25 + 1)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160, facecolor=SURFACE)
    ax.axvspan(0, 600, color="#f0efec", zorder=0)
    ax.text(300, 0.03, "length of the training examples", color="#898781", fontsize=9, ha="center", transform=ax.get_xaxis_transform())
    ax.plot(moments, activity(true), color=BLUE, linewidth=2, label="Simulation")
    ax.plot(moments, activity(predicted), color=ORANGE, linewidth=2, linestyle=(0, (6, 2)), label="Neural network")
    style(ax, "snapshots predicted in a row by the network on its own", "sideways motion behind the obstacle\n(fraction of the inflow speed)")
    ax.set_xlim(0, FRAMES)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=10, labelcolor=MUTED, loc="center right")
    fig.savefig(RESULTS / "long_run_activity.png", bbox_inches="tight")
    plt.close(fig)

    # Pictures of both at a few moments.
    shown = (100, 400, 700, 1000, 1500)
    limit = 0.8 * np.nanmax(np.abs(spin(true[-1], case["solid"])[:, CENTRE_X + 12 :]))
    cmap = plt.get_cmap("berlin").copy()
    cmap.set_bad("#8a8f98")
    fig, axes = plt.subplots(2, len(shown), figsize=(16, 2.6), dpi=160, facecolor="#101014")
    fig.subplots_adjust(left=0.085, right=0.998, top=0.88, bottom=0.01, hspace=0.05, wspace=0.02)
    for column, moment in enumerate(shown):
        for row, data in enumerate((true, predicted)):
            ax = axes[row, column]
            ax.imshow(spin(data[moment - 1], case["solid"]), cmap=cmap, vmin=-limit, vmax=limit, origin="lower", interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(f"snapshot {moment}", color="#c3c2b7", fontsize=10)
            if column == 0:
                ax.set_ylabel(("Simulation", "Neural network")[row], color="#f0efec", fontsize=10, rotation=0, ha="right", va="center", labelpad=8)
    fig.savefig(RESULTS / "long_run_snapshots.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    save_animation(true, predicted, case["solid"], RESULTS / "long_run.gif", f"Reynolds {REYNOLDS:.0f}, 1500 snapshots, with training examples of 600", every=12)
    print("network activity by snapshot:", dict(zip(moments[3::4].tolist(), np.round(activity(predicted)[3::4], 3).tolist())))
    print("simulation activity:", round(float(activity(true).mean()), 3))
