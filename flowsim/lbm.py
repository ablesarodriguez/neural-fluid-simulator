"""Two-dimensional fluid flow with the lattice Boltzmann method, on the GPU.

This is the "real" simulation: the one that produces the data the neural
network learns from, and the reference its predictions are compared with.

The method does not track the velocity of the fluid directly. On every cell of
a grid it keeps nine numbers: how much fluid is moving in each of nine
directions (staying still, the four neighbours and the four diagonals). One
time step is two operations:

  collide   the nine amounts of each cell relax towards their equilibrium
  stream    each amount moves one cell along its own direction

Density and velocity are sums of those nine numbers. With a small velocity,
this reproduces the Navier-Stokes equations of an incompressible fluid.

Several independent simulations run at once, as a batch: they share every
operation, which is much faster on a GPU than running them one after another.

Units are those of the grid: lengths in cells, times in steps.
"""

import torch

# The nine directions, as (x, y) steps on the grid, and their weights.
DIRECTIONS = [(1, 1), (1, 0), (1, -1), (0, 1), (0, 0), (0, -1), (-1, 1), (-1, 0), (-1, -1)]
WEIGHTS = [1 / 36, 1 / 9, 1 / 36, 1 / 9, 4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36]
# With this ordering, the direction opposite to i is 8 - i, and the indices
# 0-2 move right, 3-5 do not move in x and 6-8 move left.


def cylinder(nx, ny, centre_x, centre_y, radius, device="cuda"):
    """Mask (nx, ny) of the cells inside a circular obstacle."""
    x = torch.arange(nx, device=device).view(nx, 1)
    y = torch.arange(ny, device=device).view(1, ny)
    return (x - centre_x) ** 2 + (y - centre_y) ** 2 < radius**2


def square(nx, ny, centre_x, centre_y, half_side, device="cuda"):
    """Mask (nx, ny) of the cells inside a square obstacle."""
    x = torch.arange(nx, device=device).view(nx, 1)
    y = torch.arange(ny, device=device).view(1, ny)
    return ((x - centre_x).abs() < half_side) & ((y - centre_y).abs() < half_side)


class Flow:
    """Fluid entering from the left at a steady speed and flowing past an obstacle.

    obstacle    boolean tensor (nx, ny), or (batch, nx, ny) for several
                simulations at once; True where the fluid cannot go
    reynolds    Reynolds number: inflow speed * length / viscosity. It decides
                the character of the flow (smooth, shedding vortices, chaotic).
                One number, or one per simulation of the batch
    length      the size of the obstacle in cells, used in the Reynolds
                number. One number, or one per simulation
    speed       inflow speed in cells per step; it must stay well below 1
    """

    def __init__(self, obstacle, reynolds, length, speed=0.05):
        if obstacle.dim() == 2:
            obstacle = obstacle.unsqueeze(0)
        self.obstacle = obstacle
        self.device = obstacle.device
        self.batch, self.nx, self.ny = obstacle.shape
        self.speed = speed

        reynolds = torch.as_tensor(reynolds, dtype=torch.float32, device=self.device).expand(self.batch)
        length = torch.as_tensor(length, dtype=torch.float32, device=self.device).expand(self.batch)
        viscosity = speed * length / reynolds
        # How fast the cells relax to equilibrium; one value per simulation.
        self.omega = (1.0 / (3.0 * viscosity + 0.5)).view(self.batch, 1, 1, 1)

        self.c = torch.tensor(DIRECTIONS, dtype=torch.float32, device=self.device)  # (9, 2)
        self.w = torch.tensor(WEIGHTS, dtype=torch.float32, device=self.device).view(1, 9, 1, 1)

        # Inflow velocity (2, ny): uniform, towards the right.
        self.inflow = torch.zeros(2, self.ny, device=self.device)
        self.inflow[0] = speed

        # Start with the fluid already moving, plus a sideways wobble that
        # breaks the symmetry so that the vortices start shedding soon.
        x = torch.arange(self.nx, device=self.device, dtype=torch.float32).view(self.nx, 1)
        y = torch.arange(self.ny, device=self.device, dtype=torch.float32).view(1, self.ny)
        start = torch.zeros(self.batch, 2, self.nx, self.ny, device=self.device)
        start[:, 0] = speed
        start[:, 1] = 0.1 * speed * torch.sin(2.0 * torch.pi * x / self.nx) * torch.cos(2.0 * torch.pi * y / self.ny)
        density = torch.ones(self.batch, 1, self.nx, self.ny, device=self.device)
        self.f = self._equilibrium(density, start)  # (batch, 9, nx, ny)
        self.steps_done = 0

    def _equilibrium(self, density, velocity):
        """The nine amounts a cell would have at rest with this density and velocity."""
        along = 3.0 * torch.einsum("id,bdxy->bixy", self.c, velocity)  # velocity along each direction
        speed2 = 1.5 * (velocity**2).sum(dim=1, keepdim=True)
        return density * self.w * (1.0 + along + 0.5 * along**2 - speed2)

    def _macroscopic(self, f):
        density = f.sum(dim=1, keepdim=True)
        velocity = torch.einsum("id,bixy->bdxy", self.c, f) / density
        return density, velocity

    def step(self, steps=1):
        """Advances the simulation."""
        f = self.f
        solid = self.obstacle.unsqueeze(1)  # (batch, 1, nx, ny)
        for _ in range(steps):
            # Right edge: the fluid leaves freely.
            f[:, 6:9, -1, :] = f[:, 6:9, -2, :]

            density, velocity = self._macroscopic(f)

            # Left edge: the velocity is imposed, and the density follows from it.
            velocity[:, :, 0, :] = self.inflow
            density[:, 0, 0, :] = (f[:, 3:6, 0, :].sum(dim=1) + 2.0 * f[:, 6:9, 0, :].sum(dim=1)) / (1.0 - self.speed)
            equilibrium = self._equilibrium(density, velocity)
            f[:, 0:3, 0, :] = equilibrium[:, 0:3, 0, :] + f[:, [8, 7, 6], 0, :] - equilibrium[:, [8, 7, 6], 0, :]

            # Collide. On the obstacle the fluid bounces straight back instead,
            # which leaves it at rest there.
            after = torch.where(solid, f.flip(1), f - self.omega * (f - equilibrium))

            # Stream. Top and bottom are joined, as if the obstacle repeated itself.
            for i, (dx, dy) in enumerate(DIRECTIONS):
                f[:, i] = torch.roll(after[:, i], shifts=(dx, dy), dims=(1, 2))
        self.steps_done += steps

    def velocity(self):
        """Velocity of the fluid, tensor (batch, 2, nx, ny): x and y components."""
        velocity = self._macroscopic(self.f)[1]
        return velocity.masked_fill(self.obstacle.unsqueeze(1), 0.0)


def vorticity(velocity):
    """How fast the fluid spins at each cell. Positive is anticlockwise.

    velocity is (..., 2, nx, ny); the result is (..., nx, ny).
    """
    return torch.gradient(velocity[..., 1, :, :], dim=-2)[0] - torch.gradient(velocity[..., 0, :, :], dim=-1)[0]
