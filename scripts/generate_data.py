"""Runs the simulations the neural network learns from, and the ones it is tested on.

Every simulation is a fluid flowing past one obstacle. What changes between
them is the Reynolds number and the obstacle. They are split into:

  train         28 cylinders of random size and height, Reynolds 60 to 200
  test_similar  4 more of the same kind, never shown during training
  test_sweep    one fixed cylinder at Reynolds 60, 80 ... 200 and, beyond
                anything seen in training, 230 and 260
  test_square   a square obstacle, a shape never seen in training

Each one is saved in data/<split>/<name>.pt as 200 snapshots of the velocity
field, 100 simulation steps apart, at half the resolution of the simulation.

Run from the project folder:  python scripts/generate_data.py
"""

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from flowsim.lbm import Flow, cylinder, square

NX, NY, SPEED = 768, 256, 0.05     # simulation grid and inflow speed
CENTRE_X = 170                     # every obstacle sits at this distance from the inflow
SETTLE, FRAMES, EVERY = 20000, 200, 100  # steps discarded, snapshots kept, steps between snapshots
BATCH = 11                         # simulations run at the same time


def describe_all():
    """The list of simulations to run: one dictionary each."""
    random = torch.Generator().manual_seed(2026)

    def uniform(low, high):
        return low + (high - low) * torch.rand(1, generator=random).item()

    cases = []
    for split, count in (("train", 28), ("test_similar", 4)):
        for n in range(count):
            cases.append(dict(split=split, name=f"{n:02d}", shape="cylinder", reynolds=uniform(60, 200),
                              half_size=uniform(14, 22), centre_y=NY / 2 + uniform(-30, 30)))
    for reynolds in (60, 80, 100, 120, 140, 160, 180, 200, 230, 260):
        cases.append(dict(split="test_sweep", name=f"re{reynolds}", shape="cylinder", reynolds=float(reynolds),
                          half_size=18.0, centre_y=NY / 2))
    for reynolds in (100, 160):
        cases.append(dict(split="test_square", name=f"re{reynolds}", shape="square", reynolds=float(reynolds),
                          half_size=16.0, centre_y=NY / 2))
    return cases


def run(cases, device):
    """Simulates a batch of cases together and saves each one to its file."""
    masks = torch.stack([
        (cylinder if c["shape"] == "cylinder" else square)(NX, NY, CENTRE_X, c["centre_y"], c["half_size"], device)
        for c in cases
    ])
    flow = Flow(masks, [c["reynolds"] for c in cases], [2 * c["half_size"] for c in cases], SPEED)
    flow.step(SETTLE)

    frames = []
    for _ in range(FRAMES):
        flow.step(EVERY)
        # Halve the resolution by averaging 2 x 2 cells, and measure speeds in units of the inflow.
        frames.append(F.avg_pool2d(flow.velocity() / SPEED, 2).half().cpu())
    frames = torch.stack(frames, dim=1)  # (batch, frames, 2, NX / 2, NY / 2)
    solid = F.avg_pool2d(masks.float().unsqueeze(1), 2).squeeze(1).cpu()  # fraction of each cell that is obstacle

    for i, case in enumerate(cases):
        if not torch.isfinite(frames[i].float()).all():
            raise RuntimeError(f"simulation {case['split']}/{case['name']} blew up")
        folder = ROOT / "data" / case["split"]
        folder.mkdir(parents=True, exist_ok=True)
        # "diameter" is in cells of the saved grid: 2 * half_size simulation cells = half_size saved cells.
        torch.save(dict(velocity=frames[i].clone(), solid=solid[i].clone(), reynolds=case["reynolds"], shape=case["shape"],
                        diameter=case["half_size"], steps_between_frames=EVERY, inflow_speed=SPEED), folder / f"{case['name']}.pt")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cases = describe_all()
    start = time.time()
    for first in range(0, len(cases), BATCH):
        run(cases[first : first + BATCH], device)
        print(f"{min(first + BATCH, len(cases))} / {len(cases)} simulations done, {(time.time() - start) / 60:.1f} min", flush=True)
