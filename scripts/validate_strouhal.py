"""Checks the simulator against a known experimental result.

Behind a cylinder, vortices are shed at a steady rhythm. In dimensionless form
that rhythm is the Strouhal number,

    St = frequency * diameter / speed

which depends only on the Reynolds number. At Re = 100 it has been measured
many times: St = 0.164 for a cylinder alone in open flow (Williamson 1996).

Run from the project folder:  python scripts/validate_strouhal.py [height]

The optional height (default 360 cells) sets how much room the cylinder has above and
below. The reference value is for a cylinder with unlimited room; a narrow channel
squeezes the flow past it and makes the vortices shed faster.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from flowsim.lbm import Flow, cylinder

REYNOLDS, EXPECTED = 100.0, 0.164
NX, RADIUS, SPEED = 900, 15, 0.05
NY = int(sys.argv[1]) if len(sys.argv) > 1 else 360  # height of the channel, in cells
TRANSIENT, MEASURED = 30000, 40000  # steps discarded while the flow settles, then steps analysed

device = "cuda" if torch.cuda.is_available() else "cpu"
centre_x, centre_y = NX // 5, NY // 2
flow = Flow(cylinder(NX, NY, centre_x, centre_y, RADIUS, device), REYNOLDS, 2 * RADIUS, SPEED)

start = time.time()
flow.step(TRANSIENT)

# Sideways velocity at a point behind the cylinder: it swings up and down once per pair of vortices.
probe = []
for _ in range(MEASURED // 20):
    flow.step(20)
    probe.append(flow.velocity()[0, 1, centre_x + 4 * RADIUS, centre_y].item())
seconds = time.time() - start
probe = np.array(probe) - np.mean(probe)

# The frequency of the swing is the highest peak of its spectrum; a parabola through the
# peak and its two neighbours locates it more finely than the spacing of the spectrum.
spectrum = np.abs(np.fft.rfft(probe * np.hanning(len(probe))))
k = int(np.argmax(spectrum[1:])) + 1
a, b, c = np.log(spectrum[k - 1 : k + 2])
peak = k + 0.5 * (a - c) / (a - 2 * b + c)
frequency = peak / (len(probe) * 20)  # in cycles per step
strouhal = frequency * 2 * RADIUS / SPEED

print(f"device: {device}, {(TRANSIENT + MEASURED) / seconds:.0f} steps per second")
print(f"periods of shedding measured: {frequency * MEASURED:.1f}")
print(f"Strouhal number: {strouhal:.4f}   reference: {EXPECTED}   difference: {100 * (strouhal / EXPECTED - 1):+.1f} %")
