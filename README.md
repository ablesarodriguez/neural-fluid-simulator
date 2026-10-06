# Neural fluid simulator

A neural network that learns to simulate a fluid by watching simulations of it. It is never given the
equations: only examples of how a flow looks at one moment and a moment later. Left to run on its own
output, it produces the vortices shed behind an obstacle 52 times faster than the simulation it learned
from, also for obstacles and flow speeds it never saw.

![Simulation above, neural network below](docs/media/unseen_cylinder.gif)

*Above, the real simulation. Below, the network, which receives only the first frame and predicts the
other 199 by itself. This cylinder was not in the training data. Colour is the spin of the fluid: blue
one way, red the other.*

Python · [PyTorch](https://pytorch.org) · CUDA. Trained in 70 minutes on an RTX 3060.

**Status.** This is version 1: simulator, data, network, training and evaluation are complete and the
results below are measured. Version 2 is in progress, see [Roadmap](#roadmap).

## What it shows

| | |
| --- | --- |
| ![A square obstacle](docs/media/square.gif) | **A shape it never saw.** Every training example is a cylinder. Given a square, the network still produces the right wake, with an error close to that of the cylinders. |
| ![Reynolds number 260](docs/media/higher_reynolds.gif) | **A faster flow than any in training.** The training flows have Reynolds numbers from 60 to 200. At 260 the network still sheds vortices at the right rhythm, within 3.4 %. |

![Simulation, network and difference at three moments](docs/media/snapshots.png)

*The same unseen cylinder after 10, 50 and 199 predicted snapshots. The prediction stays on the real
flow for about 50 snapshots; later the vortices have the right shape but run slightly ahead or behind,
which is what the bottom row shows.*

## The method

**The simulation.** Two-dimensional flow past an obstacle, solved with the lattice Boltzmann method
(D2Q9 lattice, BGK collision) written in PyTorch so that it runs on the GPU. The fluid enters from the
left at constant speed, leaves freely on the right, and the top and bottom edges are joined. What
decides the character of the flow is the Reynolds number,

$$\mathrm{Re} = \frac{U D}{\nu}$$

with U the inflow speed, D the size of the obstacle and ν the viscosity. Between about 50 and 200 the
wake of a cylinder is a regular street of alternating vortices, the von Kármán vortex street.

**The data.** 44 simulations on a 768 × 256 grid, each saved as 200 snapshots of the velocity field,
100 simulation steps apart, at half resolution (384 × 128):

| Set | Simulations | Contents |
| --- | --- | --- |
| Training | 28 | Cylinders of random diameter and height, Reynolds 60 to 200 |
| Unseen cylinders | 4 | The same kind, never used in training |
| Reynolds sweep | 10 | One fixed cylinder at Reynolds 60, 80 … 200, and 230 and 260 |
| Square | 2 | A square obstacle at Reynolds 100 and 160 |

**The network.** A U-Net with 1.5 million parameters. Its input is the velocity field, the shape of
the obstacle and the Reynolds number; its output is the change of the velocity field over one snapshot.
Its convolutions are padded to match the simulation: joined at top and bottom, open at left and right.

**Training.** Four stages, 11 000 steps in total:

1. Predict the next snapshot.
2. Predict 4 snapshots in a row, feeding on its own output, with the error measured on all of them.
3. The same with 8 snapshots.
4. Run alone for up to 40 snapshots with no correction, and only then be trained on the 8 that follow.

The last stage shows the network the slightly wrong flows that it itself produces, so that it learns to
bring them back. It is what made the difference: without it the network was accurate for about 50
snapshots and then diverged to meaningless values; with it, it does not.

## Validation

**The simulator against experiment.** Behind a cylinder, vortices are shed at a steady rhythm. In
dimensionless form that rhythm is the Strouhal number, St = f D / U, measured many times in the
laboratory. `scripts/validate_strouhal.py` measures it in the simulator at Reynolds 100:

| Channel height | Simulator | Experiment, open flow | Difference |
| --- | --- | --- | --- |
| 12 diameters | 0.1752 | 0.164 | +6.8 % |
| 24 diameters | 0.1734 | 0.164 | +5.7 % |

The simulator sheds faster than a cylinder in open flow. Part of the reason is that its cylinder sits
in a channel of finite height, which squeezes the flow past it: with twice the room the difference
shrinks. That does not account for all of it; the rest is probably the coarse cylinder (30 cells
across), which has not been checked. The simulations in the dataset use a narrower channel still (6 to
9 diameters), so their Strouhal numbers are higher again. The network is therefore compared with the
simulator, not with open-flow experiments.

**The network against the simulator.** All the numbers below are for simulations the network never
saw, with the network running on its own from the first snapshot (`scripts/evaluate.py`).

*Shedding rhythm.* The property an engineer would ask for:

![Strouhal number against Reynolds number](docs/media/strouhal.png)

| Reynolds | 60 | 80 | 100 | 120 | 140 | 160 | 180 | 200 | 230 | 260 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Simulator | 0.155 | 0.172 | 0.184 | 0.194 | 0.200 | 0.204 | 0.208 | 0.212 | 0.217 | 0.222 |
| Network | 0.160 | 0.165 | 0.178 | 0.190 | 0.198 | 0.204 | 0.210 | 0.215 | 0.222 | 0.230 |
| Difference | +2.9 % | −4.0 % | −3.4 % | −1.8 % | −1.0 % | −0.4 % | +0.5 % | +1.1 % | +2.2 % | +3.4 % |

The last two columns are outside the range of Reynolds numbers used in training.

*Error snapshot by snapshot.* A much stricter measure: the distance between the predicted and the true
velocity field at the same instant, as a percentage of the size of the flow pattern. It punishes a
correct wake that is slightly out of step as hard as a wrong one.

![Error against the number of snapshots predicted](docs/media/rollout_error.png)

| Snapshots predicted in a row | 10 | 50 | 199 |
| --- | --- | --- | --- |
| Unseen cylinders | 11 % | 16 % | 67 % |
| Higher Reynolds (230, 260) | 22 % | 24 % | 81 % |
| Square obstacle | 10 % | 23 % | 72 % |
| No model, assuming the flow does not change | 122 % | 136 % | 100 % |

*Speed.* One snapshot takes the simulator 175 ms (100 steps on a 768 × 256 grid) and the network
3.3 ms, both on the same RTX 3060: 52 times faster. The comparison flatters the network somewhat,
because the simulator is plain PyTorch and not an optimised solver.

## Build

Requirements: Python 3.10 or newer and an NVIDIA GPU with CUDA.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

On Linux or macOS the second line is `source .venv/bin/activate`. This is the configuration the project
is developed and tested on: Windows 11, Python 3.12, PyTorch 2.6 with CUDA 12.4.

## Run

```bash
python scripts/validate_strouhal.py   # the simulator against experiment, 3 minutes
python scripts/generate_data.py       # the 44 simulations, 35 minutes, 1.7 GB in data/
python scripts/train.py               # the network, 70 minutes, saved in runs/model.pt
python scripts/evaluate.py            # every number, chart and animation above, in results/
python scripts/long_run.py            # a run five times longer than the training examples, 5 minutes
```

Times are for an RTX 3060.

## What it does not do

- **It does not last forever.** Every training example is 200 snapshots long. Left running for 2000,
  the network keeps a realistic wake for 300 to 900 snapshots (9 to 23 shedding cycles, longest at low
  Reynolds numbers) and then the vortices die out and the flow goes smooth. It no longer blows up, but
  from there on it is wrong. This is the main target of version 2.

  ![Simulation and network at five moments of a run of 1000 snapshots](docs/media/long_run_snapshots.png)

  ![Sideways motion behind the obstacle over 1000 snapshots](docs/media/long_run_activity.png)

  *Reynolds 160, a run of 1000 snapshots (`scripts/long_run.py`). The network follows the simulation
  for about 450 snapshots, more than twice the length of anything it was trained on, and then loses the
  wake within 100 more.*
- **It drifts out of step.** After about 50 snapshots the predicted vortices are in the right place for
  the wrong instant. Rhythm and shape are right; the exact timing is not.
- **It only knows this kind of flow.** One obstacle, two dimensions, Reynolds numbers of a few hundred,
  where the wake is regular. Turbulence, three dimensions or several obstacles are untested.
- **It learned from one simulator.** Its errors include those of the simulator, among them the effect
  of the narrow channel described above.
- **The training set is small.** 28 simulations. The Reynolds numbers 60 and 80 of the sweep, where the
  wake is still developing in the data, are among the worst predicted.

## Roadmap

- [x] **Version 1.** Simulator validated against experiment, dataset, U-Net, four-stage training,
  evaluation on unseen obstacles and Reynolds numbers.
- [ ] **Version 2, in progress.** Longer simulations and longer unattended runs during training, so
  that the network sustains the wake indefinitely; a larger and more varied dataset; a measure of error
  that separates being out of step from being wrong.

## Layout

| Path | Contents |
| --- | --- |
| `flowsim/lbm.py` | The lattice Boltzmann simulator |
| `flowsim/model.py` | The neural network |
| `flowsim/data.py` | Loading the simulations and the error measure |
| `scripts/validate_strouhal.py` | The simulator against the experimental Strouhal number |
| `scripts/generate_data.py` | Runs and saves the 44 simulations |
| `scripts/train.py` | Trains the network |
| `scripts/evaluate.py` | Measures the network against the simulator and draws the figures |
| `scripts/long_run.py` | A run five times longer than the training examples |
| `docs/media/` | The figures and animations of this README |

## References

- T. Krüger et al., *The Lattice Boltzmann Method: Principles and Practice*, Springer (2017).
- Q. Zou, X. He, "On pressure and velocity boundary conditions for the lattice Boltzmann BGK model",
  *Physics of Fluids* 9 (1997).
- C. H. K. Williamson, "Vortex dynamics in the cylinder wake", *Annual Review of Fluid Mechanics* 28
  (1996).
- O. Ronneberger, P. Fischer, T. Brox, "U-Net: convolutional networks for biomedical image
  segmentation", *MICCAI* (2015).
- A. Sanchez-Gonzalez et al., "Learning to simulate complex physics with graph networks", *ICML* (2020).
- J. Brandstetter, D. Worrall, M. Welling, "Message passing neural PDE solvers", *ICLR* (2022). Source
  of the idea behind the fourth training stage.
- K. Stachenfeld et al., "Learned coarse models for efficient turbulence simulation", *ICLR* (2022).

## License

MIT, see [LICENSE](LICENSE).
