"""Trains the neural network on the simulations in data/train.

Training is done in four stages. In the first the network only has to get
the next snapshot right. In the next two it must predict several snapshots in
a row, feeding on its own output, and is corrected on all of them: that is
what teaches it not to let small errors grow.

The last stage teaches it to keep going for a long time. A few predictions
are kept running throughout the stage: at every step the network advances
each of them a few snapshots further, starting from where it left off, so
that it gets to see the flows it produces after hundreds of snapshots on its
own. To correct it, each of those flows is compared with the real simulation
at the moment that looks most like it, not at the same moment by the clock.
After a long time alone the network is inevitably a little ahead or behind
the simulation, and being told off for that would only teach it to blur the
vortices away; what it has to learn is to stay on a real flow.

A fifth stage refines the result: the same as the fourth, with a small and
constant learning rate. The quality of the network swings from one check to
the next at this point, so instead of keeping whatever it is at the end, it is
measured every 250 steps on a few validation flows and the best one is kept.

With --average, a second copy of the network is kept that is never trained:
after every step each of its numbers moves a little towards the trained
network, so that it is a running average of the last thousand steps or so.
Each training step corrects the network from a handful of examples and knocks
it about a bit; the average smooths that out, and it is then the average that
is measured, saved and used afterwards. This is an experiment, off by default:
see the README for what it gained and what it did not.

Run from the project folder:  python scripts/train.py [--average] [--from-stage N] [--steps M]
The result is saved in runs/model.pt. With --from-stage N the first N - 1
stages are skipped and training continues from the saved network. With
--steps M the last stage lasts M steps instead of 3000.
"""

import copy
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from flowsim.data import DATA, load_split, relative_error
from flowsim.model import FlowNet, rollout

# (snapshots predicted in a row, batch size, training steps, kind of stage)
STAGES = [(1, 16, 2500, "fresh"), (4, 8, 3000, "fresh"), (8, 4, 2500, "fresh"), (4, 8, 6000, "running"), (4, 8, 3000, "refine")]
LEARNING_RATE = 3e-4
NOISE = 0.01          # the inputs are blurred with a little noise, so that the network learns to recover from its own errors
AVERAGING = 0.999     # how much of the averaged network is kept at each step; it then spans about the last 1000 steps
LONGEST_ALONE = 1000  # in the last stage, snapshots a prediction is kept running before it is replaced by a fresh one


def outline(velocity):
    """A coarse copy of a flow, (..., 2, nx, ny) -> (..., 1536 numbers), used to tell how alike two flows are."""
    lead = velocity.shape[:-3]
    return F.avg_pool2d(velocity.reshape(-1, *velocity.shape[-3:]).float(), 8).reshape(*lead, -1)


def check(model, cases, steps=199, long=1500):
    """Lets the network run on its own on unseen simulations.

    Returns its mean error after 10, 100 and `steps` snapshots, and in how many
    of the simulations the wake is still alive after `long` snapshots.
    """
    model.eval()
    errors, alive = [], 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for case in cases:
            true = case["velocity"][1 : steps + 1].float().unsqueeze(0)
            predicted = rollout(model, case["velocity"][:1].float(), case["solid"][None], torch.tensor([case["reynolds"]], device=true.device), long)
            errors.append(relative_error(predicted[:, :steps].float(), true)[0])
            # Sideways motion behind the obstacle at the end, against the same in the simulation.
            real, final = case["velocity"][:, 1, 105:285].float().abs().mean(), predicted[0, -50:, 1, 105:285].float().abs().mean()
            alive += int(0.5 * real < final < 2.0 * real)
    model.train()
    errors = torch.stack(errors).mean(dim=0)
    return errors[9].item(), errors[99].item(), errors[-1].item(), alive


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    device = "cuda"
    # Progress is watched, and the final network chosen, on flows that are used neither for
    # training nor for the tests of scripts/evaluate.py.
    validation = load_split("validation", device)
    # The training simulations are copied to the graphics card one at a time, straight into
    # one big array, so that they never take up twice their size in memory.
    files = sorted((DATA / "train").glob("*.pt"))
    shape = torch.load(files[0])["velocity"].shape
    velocity = torch.empty((len(files), *shape), dtype=torch.float16, device=device)  # (simulations, frames, 2, nx, ny)
    solid, reynolds, outlines = [], [], []
    for i, file in enumerate(files):
        case = torch.load(file)
        velocity[i] = case["velocity"].to(device)
        solid.append(case["solid"].to(device))
        reynolds.append(case["reynolds"])
        outlines.append(outline(velocity[i]))
    solid, outlines = torch.stack(solid), torch.stack(outlines)  # outlines: (simulations, frames, 1536)
    reynolds = torch.tensor(reynolds, device=device)
    simulations, frames = velocity.shape[:2]

    model = FlowNet().to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} million parameters, {simulations} simulations of {frames} snapshots", flush=True)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    total = sum(stage[2] for stage in STAGES if stage[3] != "refine")
    done, start = 0, time.time()
    first_stage = int(sys.argv[sys.argv.index("--from-stage") + 1]) if "--from-stage" in sys.argv else 1
    if first_stage > 1:
        model.load_state_dict(torch.load(ROOT / "runs" / "model.pt"))
        done = sum(stage[2] for stage in STAGES[: first_stage - 1])
    averaging = "--average" in sys.argv
    # The averaged network: never trained, only moved towards the trained one. Without
    # --average it is simply the trained network itself.
    average = copy.deepcopy(model) if averaging else model

    def fresh(batch, horizon):
        """Random moments of random simulations, with a little noise."""
        which = torch.randint(simulations, (batch,), device=device)
        when = torch.randint(frames - horizon, (batch,), device=device)
        state = velocity[which, when].float()
        return which, when, state + NOISE * torch.randn_like(state)

    if "--steps" in sys.argv:
        # A different length for the last stage, to keep refining a network that is still improving.
        STAGES[-1] = (*STAGES[-1][:2], int(sys.argv[sys.argv.index("--steps") + 1]), STAGES[-1][3])

    best = float("inf")
    for horizon, batch, steps, kind in STAGES[first_stage - 1 :]:
        if kind != "fresh":
            which, when, current = fresh(batch, horizon)
            alone = torch.zeros(batch, dtype=torch.long, device=device)  # snapshots each prediction has been running
        if kind == "refine":
            # The network as it enters the refinement is the one to beat.
            short, medium, long, alive = check(average, validation)
            best = (short + medium + long) / 3 if alive == len(validation) else float("inf")
            print(f"before refining: {100 * short:.1f} % after 10 snapshots, {100 * medium:.1f} % after 100, {100 * long:.1f} % after 199;  "
                  f"wake alive after 1500: {alive} of {len(validation)}", flush=True)

        for _ in range(steps):
            # The learning rate follows half a cosine, from its full value down to a twentieth of it,
            # and stays low and constant during the final refinement.
            rate = 0.1 if kind == "refine" else 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * done / total))
            for group in optimiser.param_groups:
                group["lr"] = LEARNING_RATE * rate

            if kind == "fresh":
                which, when, current = fresh(batch, horizon)
            else:
                # Replace the predictions that have run long enough (or gone astray) by fresh ones.
                old = (alone >= LONGEST_ALONE) | ~torch.isfinite(current).all(dim=(1, 2, 3))
                if old.any():
                    new_which, _, new_state = fresh(batch, horizon)
                    which = torch.where(old, new_which, which)
                    current = torch.where(old.view(-1, 1, 1, 1), new_state, current)
                    alone = torch.where(old, 0, alone)
                # For each one, the moment of its simulation that it looks most like.
                distance = (outlines[which, : frames - horizon] - outline(current).unsqueeze(1)).pow(2).sum(dim=-1)
                when = distance.argmin(dim=1)

            loss = 0.0
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for ahead in range(1, horizon + 1):
                    current = model(current, solid[which], reynolds[which])
                    loss = loss + (current.float() - velocity[which, when + ahead].float()).pow(2).mean()
            loss = loss / horizon

            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            # Early on the average follows the network closely, so that it is not held back by the random start.
            keep = min(AVERAGING, (1 + done) / (10 + done))
            if averaging:
                with torch.no_grad():
                    for averaged, trained in zip(average.parameters(), model.parameters()):
                        averaged.lerp_(trained, 1.0 - keep)
            done += 1
            if kind != "fresh":
                current = current.detach().float()  # the next step continues from here
                alone += horizon

            if done % (250 if kind == "refine" else 500) == 0:
                short, medium, long, alive = check(average, validation)
                (ROOT / "runs").mkdir(exist_ok=True)
                torch.save(average.state_dict(), ROOT / "runs" / "last.pt")
                kept = ""
                if kind != "refine":
                    torch.save(average.state_dict(), ROOT / "runs" / "model.pt")
                elif alive == len(validation) and (short + medium + long) / 3 < best:
                    # During the refinement only the best network so far is kept: the one with the
                    # lowest error among those that keep the wake alive in every validation flow.
                    best = (short + medium + long) / 3
                    torch.save(average.state_dict(), ROOT / "runs" / "model.pt")
                    kept = "  <- best so far, kept"
                print(f"step {done:5d}  {kind}, {horizon} ahead  loss {loss.item():.2e}  "
                      f"error on validation flows: {100 * short:.1f} % after 10 snapshots, {100 * medium:.1f} % after 100, {100 * long:.1f} % after 199;  "
                      f"wake alive after 1500: {alive} of {len(validation)}  ({(time.time() - start) / 60:.1f} min){kept}", flush=True)

    print("the network is in runs/model.pt")
