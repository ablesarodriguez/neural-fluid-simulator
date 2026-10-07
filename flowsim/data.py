"""Loading the simulations saved by scripts/generate_data.py."""

import os
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
# The simulations are read from data/, or from the folder named in FLOWSIM_DATA.
DATA = ROOT / os.environ.get("FLOWSIM_DATA", "data")


def load_split(split, device="cuda"):
    """Returns the simulations of one split as a list of dictionaries:

    velocity  (frames, 2, nx, ny) in units of the inflow speed
    solid     (nx, ny) fraction of each cell taken by the obstacle
    reynolds, diameter (in cells), shape, name
    """
    folder = DATA / split
    files = sorted(folder.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"no simulations in {folder}; run scripts/generate_data.py first")
    cases = []
    for file in files:
        case = torch.load(file)
        case["velocity"] = case["velocity"].to(device)  # kept as 16-bit floats to save memory
        case["solid"] = case["solid"].to(device)
        case["name"] = file.stem
        cases.append(case)
    return cases


def relative_error(predicted, true):
    """How far a prediction is from the truth, per frame.

    Both are (..., 2, nx, ny). The distance is divided by the size of what
    there is to predict: the departure of the true flow from the plain uniform
    stream it would be with no obstacle. 0 is perfect; 1 means the prediction
    is as wrong as answering "nothing happens".
    """
    stream = torch.zeros_like(true)
    stream[..., 0, :, :] = 1.0
    wrong = (predicted - true).pow(2).sum(dim=(-3, -2, -1)).sqrt()
    size = (true - stream).pow(2).sum(dim=(-3, -2, -1)).sqrt()
    return wrong / size
