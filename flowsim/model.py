"""The neural network that learns to imitate the simulation.

It receives the velocity of the fluid at one moment and returns the velocity
one snapshot later (100 simulation steps). Applying it again and again to its
own output produces a whole simulation, without solving any equation.

It is a U-Net: a stack of convolutions that looks at the flow at several
scales. Fine layers see the details next to each cell; coarse layers, which
work on a shrunken copy of the image, see whole vortices. The network is never
given the equations of the fluid, only examples.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# The Reynolds number is given to the network as an extra input, rescaled
# with these so that the training range (60 to 200) maps to about -1 to 1.
REYNOLDS_CENTRE, REYNOLDS_SPREAD = 130.0, 70.0


def pad(x):
    """Adds a one-cell border, matching the edges of the simulation: the top
    and bottom are joined to each other, the left and right are not."""
    x = F.pad(x, (1, 1, 0, 0), mode="circular")   # last axis is y
    return F.pad(x, (0, 0, 1, 1), mode="replicate")  # the one before is x


class Block(nn.Module):
    """Two 3 x 3 convolutions."""

    def __init__(self, inputs, outputs):
        super().__init__()
        self.first = nn.Conv2d(inputs, outputs, 3)
        self.second = nn.Conv2d(outputs, outputs, 3)

    def forward(self, x):
        x = F.gelu(self.first(pad(x)))
        return F.gelu(self.second(pad(x)))


class FlowNet(nn.Module):
    def __init__(self, widths=(32, 64, 128, 192)):
        super().__init__()
        # Going down: each level halves the resolution and widens the features.
        self.down = nn.ModuleList()
        inputs = 4  # velocity x, velocity y, obstacle, Reynolds number
        for width in widths:
            self.down.append(Block(inputs, width))
            inputs = width
        # Going up: each level doubles the resolution and is joined with the
        # features of the same size from the way down.
        self.up = nn.ModuleList()
        for width in reversed(widths[:-1]):
            self.up.append(Block(inputs + width, width))
            inputs = width
        self.out = nn.Conv2d(inputs, 2, 1)

    def forward(self, velocity, solid, reynolds):
        """One snapshot forward in time.

        velocity  (batch, 2, nx, ny), in units of the inflow speed
        solid     (batch, nx, ny), fraction of each cell taken by the obstacle
        reynolds  (batch,)
        """
        batch, _, nx, ny = velocity.shape
        solid = solid.unsqueeze(1)
        condition = ((reynolds - REYNOLDS_CENTRE) / REYNOLDS_SPREAD).view(batch, 1, 1, 1).expand(batch, 1, nx, ny)
        x = torch.cat([velocity, solid, condition], dim=1)

        kept = []
        for level, block in enumerate(self.down):
            if level > 0:
                x = F.avg_pool2d(x, 2)
            x = block(x)
            kept.append(x)
        kept.pop()
        for block in self.up:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = block(torch.cat([x, kept.pop()], dim=1))

        # The network predicts the change, which is added to the input. Inside
        # the obstacle the fluid does not move.
        return (velocity + self.out(x)) * (1.0 - solid)


def rollout(model, velocity, solid, reynolds, steps):
    """Applies the network to its own output, returning (batch, steps, 2, nx, ny)."""
    frames = []
    for _ in range(steps):
        velocity = model(velocity, solid, reynolds)
        frames.append(velocity)
    return torch.stack(frames, dim=1)
