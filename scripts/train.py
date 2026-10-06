"""Trains the neural network on the simulations in data/train.

Training is done in four stages. In the first the network only has to get
the next snapshot right. In the next two it must predict several snapshots in
a row, feeding on its own output, and is corrected on all of them: that is
what teaches it not to let small errors grow.

In the last stage the network is first left to run on its own for a random
number of snapshots, with no correction, and only then trained on the ones
that follow. This shows it the slightly wrong flows that it itself produces
after a while, and teaches it to bring them back to the real ones. Without
this stage it is accurate for about 50 snapshots and then falls apart.

Run from the project folder:  python scripts/train.py [--from-stage N]
The result is saved in runs/model.pt. With --from-stage N the first N - 1
stages are skipped and training continues from the saved network.
"""

import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from flowsim.data import load_split, relative_error
from flowsim.model import FlowNet, rollout

# (snapshots predicted in a row, batch size, training steps, most snapshots run alone beforehand)
STAGES = [(1, 16, 2500, 0), (4, 8, 3000, 0), (8, 4, 2500, 0), (8, 4, 3000, 40)]
LEARNING_RATE = 3e-4
NOISE = 0.01  # the inputs are blurred with a little noise, so that the network learns to recover from its own errors


def check(model, cases, steps=199):
    """Lets the network run on its own on unseen simulations and returns its mean error after 10, 100 and `steps` snapshots."""
    model.eval()
    errors = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for case in cases:
            true = case["velocity"][1 : steps + 1].float().unsqueeze(0)
            predicted = rollout(model, case["velocity"][:1].float(), case["solid"][None], torch.tensor([case["reynolds"]], device=true.device), steps)
            errors.append(relative_error(predicted.float(), true)[0])
    model.train()
    errors = torch.stack(errors).mean(dim=0)
    return errors[9].item(), errors[99].item(), errors[-1].item()


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    device = "cuda"
    train, held_out = load_split("train", device), load_split("test_similar", device)
    velocity = torch.stack([case["velocity"] for case in train])  # (simulations, frames, 2, nx, ny)
    solid = torch.stack([case["solid"] for case in train])
    reynolds = torch.tensor([case["reynolds"] for case in train], device=device)
    simulations, frames = velocity.shape[:2]

    model = FlowNet().to(device)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} million parameters, {simulations} simulations of {frames} snapshots")
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    total = sum(stage[2] for stage in STAGES)
    done, start = 0, time.time()
    first_stage = int(sys.argv[sys.argv.index("--from-stage") + 1]) if "--from-stage" in sys.argv else 1
    if first_stage > 1:
        model.load_state_dict(torch.load(ROOT / "runs" / "model.pt"))
        done = sum(stage[2] for stage in STAGES[: first_stage - 1])

    for horizon, batch, steps, lead in STAGES[first_stage - 1 :]:
        for _ in range(steps):
            # The learning rate follows half a cosine, from its full value down to a twentieth of it.
            for group in optimiser.param_groups:
                group["lr"] = LEARNING_RATE * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * done / total)))

            # A random moment of a random simulation, and the snapshots that follow it.
            which = torch.randint(simulations, (batch,), device=device)
            when = torch.randint(frames - horizon - lead, (batch,), device=device)
            current = velocity[which, when].float()
            current = current + NOISE * torch.randn_like(current)
            if lead > 0:
                # Let the network run alone for a while; what it is trained on starts from there.
                alone = int(torch.randint(lead + 1, (1,)).item())
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for _ in range(alone):
                        current = model(current, solid[which], reynolds[which])
                current = current.float()
                when = when + alone
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
            done += 1

            if done % 500 == 0:
                short, medium, long = check(model, held_out)
                print(f"step {done:5d} / {total}  predicting {horizon} ahead  loss {loss.item():.2e}  "
                      f"error on unseen flows: {100 * short:.1f} % after 10 snapshots, {100 * medium:.1f} % after 100, {100 * long:.1f} % after 199  "
                      f"({(time.time() - start) / 60:.1f} min)", flush=True)

    (ROOT / "runs").mkdir(exist_ok=True)
    torch.save(model.state_dict(), ROOT / "runs" / "model.pt")
    print("saved runs/model.pt")
