"""Compares the trained network with the real simulation on flows it never saw.

It answers four questions and writes the answers to results/:

  1. How wrong is the network as it runs on its own, snapshot after snapshot?
  2. Does it get the physics right? Measured with the rhythm at which vortices
     are shed (Strouhal number), inside and outside the range it was trained on.
  3. Does it stay stable when left running far longer than any training example?
  4. How much faster is it than the simulation?

Run from the project folder:  python scripts/evaluate.py

The network is read from runs/model.pt and the results go to results/. Both can be
changed with the environment variables FLOWSIM_MODEL and FLOWSIM_RESULTS, which is
how an older network is measured on the same test simulations.
"""

import json
import os
import sys
import time
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
from flowsim.data import load_split, relative_error
from flowsim.lbm import Flow, cylinder, vorticity
from flowsim.model import FlowNet, rollout

RESULTS = ROOT / os.environ.get("FLOWSIM_RESULTS", "results")
MODEL = ROOT / os.environ.get("FLOWSIM_MODEL", "runs/model.pt")
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#898781"
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
CENTRE_X = 85  # position of every obstacle along the flow, in cells of the saved grid


def predict(model, case, steps):
    """The network running on its own from the first snapshot of a simulation: (steps, 2, nx, ny)."""
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        frames = rollout(model, case["velocity"][:1].float(), case["solid"][None],
                         torch.tensor([case["reynolds"]], device="cuda"), steps)
    return frames[0].float()


def error_without_timing(predicted, true):
    """The error of each predicted frame against the true frame that looks most like it.

    The ordinary error compares both at the same instant, so a correct wake that
    runs a little ahead or behind counts as wrong. This one lets each predicted
    frame be matched with any moment of the simulation: what is left is how far
    the prediction is from being a real flow at all.
    """
    coarse = lambda v: F.avg_pool2d(v, 8).flatten(1)
    nearest = torch.cdist(coarse(predicted), coarse(true)).argmin(dim=1)
    return relative_error(predicted, true[nearest])


def at(curve, moments=(10, 50, 199, 599)):
    """Values of an error curve, in percent, after some numbers of snapshots."""
    return {f"after_{m}": float(100 * curve[m - 1]) for m in moments if m <= len(curve)}


def strouhal(velocity, case):
    """Shedding rhythm, from the sideways velocity at a point two diameters behind the obstacle."""
    solid = case["solid"]
    centre_y = int(round((solid.sum(dim=0) * torch.arange(solid.shape[1], device=solid.device)).sum().item() / solid.sum().item()))
    probe = velocity[:, 1, CENTRE_X + int(2 * case["diameter"]), centre_y].cpu().numpy()
    probe = probe - probe.mean()
    spectrum = np.abs(np.fft.rfft(probe * np.hanning(len(probe))))
    k = int(np.argmax(spectrum[1:-1])) + 1
    a, b, c = np.log(spectrum[k - 1 : k + 2])
    cycles_per_frame = (k + 0.5 * (a - c) / (a - 2 * b + c)) / len(probe)
    # In simulation units: frequency per step * diameter in simulation cells / inflow speed.
    return cycles_per_frame / case["steps_between_frames"] * (2 * case["diameter"]) / case["inflow_speed"]


def style(ax, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=GREY, labelsize=10)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=11)
    ax.set_ylabel(ylabel, color=MUTED, fontsize=11)


def spin(velocity, solid):
    """Vorticity as a picture: (ny, nx) array with the obstacle blanked out."""
    w = vorticity(velocity)
    w = torch.where(solid > 0.5, torch.nan, w)
    return w.T.cpu().numpy()


def save_animation(true, predicted, solid, path, title, every=3):
    """The simulation above and the network below, side by side in time."""
    frames = range(0, true.shape[0], every)
    limit = 0.8 * np.nanmax(np.abs(spin(true[-1], solid)[:, CENTRE_X + 12 :]))
    cmap = plt.get_cmap("berlin").copy()
    cmap.set_bad("#8a8f98")
    fig, axes = plt.subplots(2, 1, figsize=(7.68, 5.5), dpi=75, facecolor="#101014")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, hspace=0.12)
    images = []
    for ax, data, label in zip(axes, (true, predicted), ("Simulation", "Neural network")):
        ax.axis("off")
        ax.set_title(label, color="#f0efec", fontsize=11, loc="left", pad=4)
        images.append(ax.imshow(spin(data[0], solid), cmap=cmap, vmin=-limit, vmax=limit, origin="lower", interpolation="bilinear"))
    fig.suptitle(title, color="#c3c2b7", fontsize=10, x=0.99, ha="right")
    counter = fig.text(0.01, 0.965, "", color="#c3c2b7", fontsize=10)

    def update(k):
        images[0].set_data(spin(true[k], solid))
        images[1].set_data(spin(predicted[k], solid))
        counter.set_text(f"snapshot {k + 1}")

    FuncAnimation(fig, update, frames=frames).save(path, writer=PillowWriter(fps=12))
    plt.close(fig)


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    model = FlowNet().cuda()
    model.load_state_dict(torch.load(MODEL))
    model.eval()
    summary = {}

    # ---- 1. Error of the network running on its own ----------------------------------
    groups = {
        "Unseen cylinders, Reynolds 60 to 200": load_split("test_similar"),
        "Higher Reynolds than in training (230, 260)": [c for c in load_split("test_sweep") if c["reynolds"] > 200],
        "Square obstacle, a shape never seen": load_split("test_square"),
    }
    curves, shapes, kept = {}, {}, {}
    for name, cases in groups.items():
        errors, untimed = [], []
        for case in cases:
            true = case["velocity"][1:].float()
            predicted = predict(model, case, true.shape[0])
            errors.append(relative_error(predicted, true).cpu().numpy())
            untimed.append(error_without_timing(predicted, true).cpu().numpy())
            kept[(name, case["name"])] = (case, true, predicted)
        curves[name] = np.mean(errors, axis=0)
        shapes[name] = np.mean(untimed, axis=0)
    # What you get with no model at all: assuming the flow stays as it was at the start.
    still = np.mean([relative_error(c["velocity"][:1].float().expand_as(c["velocity"][1:]), c["velocity"][1:].float()).cpu().numpy()
                     for c in groups["Unseen cylinders, Reynolds 60 to 200"]], axis=0)

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160, facecolor=SURFACE)
    snapshots = np.arange(1, len(still) + 1)
    ax.plot(snapshots, 100 * still, color=GREY, linewidth=2, linestyle=(0, (4, 3)), label="No model: assume the flow does not change")
    for (name, curve), colour, dashes in zip(curves.items(), (BLUE, ORANGE, AQUA), ("solid", (0, (6, 2)), (0, (1, 1.5)))):
        ax.plot(snapshots, 100 * curve, color=colour, linewidth=2, linestyle=dashes, label=name)
    style(ax, "snapshots predicted in a row by the network on its own", "error (% of the size of the flow pattern)")
    ax.set_xlim(0, len(still))
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=MUTED, loc="upper left", bbox_to_anchor=(0.0, 1.16), ncol=2)
    fig.savefig(RESULTS / "rollout_error.png", bbox_inches="tight")
    plt.close(fig)
    summary["error_percent"] = {name: at(c) for name, c in curves.items()}
    summary["error_percent"]["No model"] = at(still)

    # The same with the timing taken out: each predicted frame against the most similar true one.
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160, facecolor=SURFACE)
    for (name, curve), colour, dashes in zip(shapes.items(), (BLUE, ORANGE, AQUA), ("solid", (0, (6, 2)), (0, (1, 1.5)))):
        ax.plot(snapshots, 100 * curve, color=colour, linewidth=2, linestyle=dashes, label=name)
    style(ax, "snapshots predicted in a row by the network on its own", "error against the most similar true frame (%)")
    ax.set_xlim(0, len(still))
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=9.5, labelcolor=MUTED, loc="upper left", bbox_to_anchor=(0.0, 1.16), ncol=2)
    fig.savefig(RESULTS / "error_without_timing.png", bbox_inches="tight")
    plt.close(fig)
    summary["error_without_timing_percent"] = {name: at(c) for name, c in shapes.items()}

    # ---- 2. Shedding rhythm against the Reynolds number ------------------------------
    sweep = sorted(load_split("test_sweep"), key=lambda c: c["reynolds"])
    table = []
    for case in sweep:
        true = case["velocity"].float()
        predicted = torch.cat([true[:1], predict(model, case, true.shape[0] - 1)])
        table.append((case["reynolds"], strouhal(true, case), strouhal(predicted, case)))
    table = np.array(table)

    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160, facecolor=SURFACE)
    ax.axvspan(60, 200, color="#f0efec", zorder=0)
    ax.text(130, 0.02, "range of Reynolds numbers seen in training", color=GREY, fontsize=9.5, ha="center", transform=ax.get_xaxis_transform())
    ax.plot(table[:, 0], table[:, 1], color=BLUE, linewidth=2, marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label="Simulation")
    ax.plot(table[:, 0], table[:, 2], color=ORANGE, linewidth=0, marker="D", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2, label="Neural network")
    style(ax, "Reynolds number", "Strouhal number (rhythm of vortex shedding)")
    ax.margins(y=0.25)
    ax.legend(frameon=False, fontsize=10, labelcolor=MUTED, loc="upper left")
    fig.savefig(RESULTS / "strouhal.png", bbox_inches="tight")
    plt.close(fig)
    summary["strouhal"] = [{"reynolds": float(r), "simulation": float(s), "network": float(n), "difference_percent": float(100 * (n / s - 1))} for r, s, n in table]

    # ---- 3. How long the network keeps going, far beyond the length of the training examples ----
    # Every training example is 200 snapshots long. Here the network runs for 2000. The flow
    # is watched through how much sideways motion there is behind the obstacle: the network
    # is counted as "lasting" until that falls below half of what the simulation has.
    long_runs = []
    wake = slice(CENTRE_X + 20, CENTRE_X + 200)
    for case in sweep:
        true = case["velocity"].float()
        predicted = predict(model, case, 2000)
        real_level = true[:, 1, wake].abs().mean().item()
        levels = predicted[:, 1, wake].abs().mean(dim=(1, 2)).unfold(0, 50, 50).mean(dim=1).cpu().numpy()  # in windows of 50
        alive = np.nonzero((levels < 0.5 * real_level) | (levels > 2.0 * real_level))[0]
        lasts = int(50 * alive[0]) if len(alive) else 2000
        frames_per_cycle = 1.0 / (strouhal(true, case) * case["steps_between_frames"] * case["inflow_speed"] / (2 * case["diameter"]))
        long_runs.append({"reynolds": case["reynolds"], "finite": bool(torch.isfinite(predicted).all()), "lasts_snapshots": lasts,
                          "lasts_shedding_cycles": lasts / frames_per_cycle, "reached_the_end": lasts == 2000})
    summary["long_run"] = long_runs

    # ---- 4. Speed --------------------------------------------------------------------
    case = sweep[2]
    flow = Flow(cylinder(768, 256, 170, 128, 18), 100.0, 36.0)
    flow.step(50)
    torch.cuda.synchronize()
    start = time.time()
    flow.step(300)
    torch.cuda.synchronize()
    simulation_ms = 1000 * (time.time() - start) / 3  # per snapshot of 100 steps
    velocity, solid, reynolds = case["velocity"][:1].float(), case["solid"][None], torch.tensor([100.0], device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(10):
            model(velocity, solid, reynolds)
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(100):
            velocity = model(velocity, solid, reynolds)
        torch.cuda.synchronize()
    network_ms = 1000 * (time.time() - start) / 100
    summary["speed"] = {"simulation_ms_per_snapshot": simulation_ms, "network_ms_per_snapshot": network_ms, "speed_up": simulation_ms / network_ms}

    # ---- Pictures ----------------------------------------------------------------------
    case, true, predicted = kept[("Unseen cylinders, Reynolds 60 to 200", "00")]
    moments = [m for m in (9, 49, 198, 598) if m < true.shape[0]]
    limit = 0.8 * np.nanmax(np.abs(spin(true[-1], case["solid"])[:, CENTRE_X + 12 :]))
    cmap = plt.get_cmap("berlin").copy()
    cmap.set_bad("#8a8f98")
    fig, axes = plt.subplots(3, len(moments), figsize=(1.6 + 3.5 * len(moments), 4.3), dpi=160, facecolor="#101014")
    fig.subplots_adjust(left=1.45 / (1.6 + 3.5 * len(moments)), right=0.995, top=0.93, bottom=0.01, hspace=0.04, wspace=0.02)
    for column, moment in enumerate(moments):
        rows = (spin(true[moment], case["solid"]), spin(predicted[moment], case["solid"]))
        for row, picture in enumerate(rows + (rows[1] - rows[0],)):
            ax = axes[row, column]
            ax.imshow(picture, cmap=cmap, vmin=-limit, vmax=limit, origin="lower", interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(f"after {moment + 1} snapshots", color="#c3c2b7", fontsize=10)
            if column == 0:
                ax.set_ylabel(("Simulation", "Neural network", "Difference")[row], color="#f0efec", fontsize=10, rotation=0, ha="right", va="center", labelpad=8)
    fig.savefig(RESULTS / "snapshots.png", facecolor=fig.get_facecolor())
    plt.close(fig)

    save_animation(true, predicted, case["solid"], RESULTS / "unseen_cylinder.gif", f"Unseen cylinder, Reynolds {case['reynolds']:.0f}", every=5)
    case, true, predicted = kept[("Higher Reynolds than in training (230, 260)", "re260")]
    save_animation(true, predicted, case["solid"], RESULTS / "higher_reynolds.gif", "Reynolds 260, above the training range (60 to 200)", every=5)
    case, true, predicted = kept[("Square obstacle, a shape never seen", "re160")]
    save_animation(true, predicted, case["solid"], RESULTS / "square.gif", "Square obstacle, a shape never seen in training", every=5)

    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
