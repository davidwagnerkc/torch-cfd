# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modifications copyright (C) 2024 S.Cao
# ported Google's Jax-CFD functional template to PyTorch's tensor ops

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from torch_cfd import grids


Grid = grids.Grid
GridVariable = grids.GridVariable
GridVariableVector = grids.GridVariableVector
ScalarField = Union[GridVariable, torch.Tensor]
VectorField = Union[GridVariableVector, Tuple[torch.Tensor, torch.Tensor]]


def forcing_eval(eval_func):
    """
    A decorator for forcing evaluators.
    This decorator simplifies the conversion of a standalone forcing evaluation function
    to a method that can be called on a class instance. It standardizes the interface
    for forcing functions by ensuring they accept grid and field parameters.
    Parameters
    ----------
    eval_func : callable
        The forcing evaluation function to be decorated. Should accept grid and field parameters
        and return a torch.Tensor representing the forcing term.
    Returns
    -------
    callable
        A wrapper function that can be used as a class method for evaluating forcing terms.
        The wrapper maintains the same signature as the decorated function but ignores the
        class instance (self) parameter.
    Examples
    --------
    @forcing_eval
    def constant_forcing(field, grid):
        return torch.ones_like(field)
    """

    def wrapper(
        cls,
        grid: Grid,
        field: Optional[Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]],
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return eval_func(grid, field)

    return wrapper


class ForcingFn(nn.Module):
    """
    A meta class for forcing functions

    Args:
    vorticity: whether the forcing function is a vorticity forcing

    Notes:
    - the grid variable is the first argument in the __call__ so that the second variable can be velocity or vorticity
    - forcing term does not have boundary conditions, when being evaluated, it is simply added to the velocity or vorticity (with the same grid)

    TODO:
    - [x] MAC grid the components of velocity does not live on the same grid.
    """

    def __init__(
        self,
        grid: Grid,
        scale: float = 1.0,
        wave_number: int = 1,
        diam: float = 1.0,
        swap_xy: bool = False,
        vorticity: bool = False,
        offsets: Optional[Tuple[Tuple[float, ...], ...]] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        super().__init__()
        self.grid = grid
        self.scale = scale
        self.wave_number = wave_number
        self.diam = diam
        self.swap_xy = swap_xy
        self.vorticity = vorticity
        self.offsets = grid.cell_faces if offsets is None else offsets
        self.device = grid.device if device is None else device

    @forcing_eval
    def velocity_eval(
        self, grid: Grid, velocity: Optional[VectorField], time: Optional[float]
    ) -> VectorField:
        raise NotImplementedError

    @forcing_eval
    def vorticity_eval(
        self, grid: Grid, vorticity: Optional[ScalarField], time: Optional[float]
    ) -> ScalarField:
        raise NotImplementedError

    def forward(
        self,
        grid: Optional[Grid] = None,
        velocity: Optional[VectorField] = None,
        vorticity: Optional[ScalarField] = None,
        time: Optional[float] = None,
    ) -> Union[ScalarField, VectorField]:
        if grid is None:
            grid = self.grid
        if not self.vorticity:
            return self.velocity_eval(grid, velocity, time)
        else:
            return self.vorticity_eval(grid, vorticity, time)


class KolmogorovForcing(ForcingFn):
    """
    The Kolmogorov forcing function used in
    Sets up the flow that is used in Kochkov et al. [1].
    which is based on Boffetta et al. [2].

    Note in the port: this forcing belongs a larger class
    of isotropic turbulence. See [3].

    References:
    [1] Machine learning-accelerated computational fluid dynamics. Dmitrii
    Kochkov, Jamie A. Smith, Ayya Alieva, Qing Wang, Michael P. Brenner, Stephan
    Hoyer Proceedings of the National Academy of Sciences May 2021, 118 (21)
    e2101784118; DOI: 10.1073/pnas.2101784118.
    https://doi.org/10.1073/pnas.2101784118

    [2] Boffetta, Guido, and Robert E. Ecke. "Two-dimensional turbulence."
    Annual review of fluid mechanics 44 (2012): 427-451.
    https://doi.org/10.1146/annurev-fluid-120710-101240

    [3] McWilliams, J. C. (1984). "The emergence of isolated coherent vortices
    in turbulent flow". Journal of Fluid Mechanics, 146, 21-43.
    """

    def __init__(
        self,
        *args,
        diam=2 * torch.pi,
        offsets=((0, 0), (0, 0)),
        vorticity=False,
        wave_number=1,
        **kwargs,
    ):
        super().__init__(
            *args,
            diam=diam,
            offsets=offsets,
            vorticity=vorticity,
            wave_number=wave_number,
            **kwargs,
        )
        self.cached = None

    def velocity_eval(
        self,
        grid: Optional[Grid],
        velocity: Optional[VectorField] = None,
        time: Optional[float] = None,
    ) -> GridVariableVector:
        if self.cached:
            return self.cached
        offsets = self.offsets
        grid = self.grid if grid is None else grid
        domain_factor = 2 * torch.pi / self.diam

        if self.swap_xy:
            x = grid.mesh(offsets[1])[0]
            v = GridVariable(
                self.scale * torch.sin(self.wave_number * domain_factor * x),
                offsets[1],
                grid,
            )
            u = GridVariable(torch.zeros_like(v.data), (1, 1 / 2), grid)
        else:
            y = grid.mesh(offsets[0])[1]
            u = GridVariable(
                self.scale * torch.sin(self.wave_number * domain_factor * y),
                offsets[0],
                grid,
            )
            v = GridVariable(torch.zeros_like(u.data), (1 / 2, 1), grid)
        self.cached = GridVariableVector(tuple((u, v)))
        return self.cached 

    def vorticity_eval(
        self,
        grid: Optional[Grid],
        vorticity: Optional[ScalarField] = None,
        time: Optional[float] = None,
    ) -> GridVariable:
        offsets = self.offsets
        grid = self.grid if grid is None else grid
        domain_factor = 2 * torch.pi / self.diam

        if self.swap_xy:
            x = grid.mesh(offsets[1])[0]
            w = GridVariable(
                -self.scale
                * self.wave_number
                * domain_factor
                * torch.cos(self.wave_number * domain_factor * x),
                offsets[1],
                grid,
            )
        else:
            y = grid.mesh(offsets[0])[1]
            w = GridVariable(
                -self.scale
                * self.wave_number
                * domain_factor
                * torch.cos(self.wave_number * domain_factor * y),
                offsets[0],
                grid,
            )
        return w


def scalar_potential(potential_func):
    def wrapper(
        cls,
        x: torch.Tensor,
        y: torch.Tensor,
        s: float,
        k: float,
        t: Optional[float] = None,
    ) -> torch.Tensor:
        return potential_func(x, y, s, k, t)

    return wrapper


class SimpleSolenoidalForcing(ForcingFn):
    """
    A simple solenoidal (rotating, divergence free) forcing function template.
    The template forcing is F = (-psi, psi) such that

    Args:
    grid: grid on which to simulate the flow
    scale: a in the equation above, amplitude of the forcing
    k: k in the equation above, wavenumber of the forcing
    """

    def __init__(
        self,
        scale=1.0,
        diam=1.0,
        wave_number=1,
        offsets=((0, 0), (0, 0)),
        vorticity=True,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            scale=scale,
            diam=diam,
            wave_number=wave_number,
            offsets=offsets,
            vorticity=vorticity,
            **kwargs,
        )

    @scalar_potential
    def potential(*args, **kwargs) -> ScalarField:
        raise NotImplementedError

    @scalar_potential
    def vort_potential(*args, **kwargs) -> ScalarField:
        raise NotImplementedError

    def velocity_eval(
        self,
        grid: Optional[Grid],
        velocity: Optional[VectorField] = None,
        time: Optional[float] = None,
    ) -> VectorField:
        offsets = self.offsets
        grid = self.grid if grid is None else grid
        domain_factor = 2 * torch.pi / self.diam
        k = self.wave_number * domain_factor
        scale = 0.5 * self.scale / (2 * torch.pi) / self.wave_number

        if self.swap_xy:
            x = grid.mesh(offsets[1])[0]
            y = grid.mesh(offsets[0])[1]
            rot = self.potential(x, y, scale, k, time)
            v = GridVariable(rot.data, offsets[1], grid)
            u = GridVariable(-rot.data, (1, 1 / 2), grid)
        else:
            x = grid.mesh(offsets[0])[0]
            y = grid.mesh(offsets[1])[1]
            rot = self.potential(x, y, scale, k, time)
            u = GridVariable(rot.data, offsets[0], grid)
            v = GridVariable(-rot.data, (1 / 2, 1), grid)
        return GridVariableVector(tuple((u, v)))

    def vorticity_eval(
        self,
        grid: Optional[Grid],
        vorticity: Optional[ScalarField] = None,
        time: Optional[float] = None,
    ) -> ScalarField:
        offsets = self.offsets
        grid = self.grid if grid is None else grid
        domain_factor = 2 * torch.pi / self.diam
        k = self.wave_number * domain_factor
        scale = self.scale

        if self.swap_xy:
            x = grid.mesh(offsets[1])[0]
            y = grid.mesh(offsets[0])[1]
            return self.vort_potential(x, y, scale, k, time)
        else:
            x = grid.mesh(offsets[0])[0]
            y = grid.mesh(offsets[1])[1]
            return self.vort_potential(x, y, scale, k, time)


class SinCosForcing(SimpleSolenoidalForcing):
    """
    The solenoidal (divergence free) forcing function used in [4].

    Note: in the vorticity-streamfunction formulation, the forcing
    is actually the curl of the velocity field, which
    is a*(sin(2*pi*k*(x+y)) + cos(2*pi*k*(x+y)))
    a=0.1, k=1 in [4]

    References:
    [4] Li, Zongyi, et al. "Fourier Neural Operator for
    Parametric Partial Differential Equations."
    ICLR. 2020.

    Args:
    grid: grid on which to simulate the flow
    scale: a in the equation above, amplitude of the forcing
    k: k in the equation above, wavenumber of the forcing
    """

    def __init__(
        self,
        scale=0.1,
        diam=1.0,
        wave_number=1,
        offsets=((0, 0), (0, 0)),
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            scale=scale,
            diam=diam,
            wave_number=wave_number,
            offsets=offsets,
            **kwargs,
        )

    def potential(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        s: float,
        k: float,
        t: Optional[float] = 0.0,
    ) -> torch.Tensor:
        return s * (torch.sin(k * (x + y)) - torch.cos(k * (x + y)))

    def vort_potential(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        s: float,
        k: float,
        t: Optional[float] = 0.0,
    ) -> torch.Tensor:
        return s * (torch.cos(k * (x + y)) + torch.sin(k * (x + y)))


class PressureGradientForcing(ForcingFn):
    """
    Uniform pressure gradient forcing for velocity fields.

    This forcing applies a constant pressure gradient across the domain,
    resulting in a uniform body force. Commonly used for channel flow
    simulations where the pressure gradient drives the flow.

    The forcing vector is (pressure_gradient, 0) in 2D, where the first
    component drives flow in the x-direction and the second component
    is zero (no y-direction pressure gradient).

    Args:
        pressure_gradient: magnitude of the pressure gradient in x-direction
        force_vector: optional custom force vector. If None, defaults to
                     (pressure_gradient, 0) for 2D
        grid: computational grid
        device: torch device for computations

    Example:
        # Create pressure gradient forcing for channel flow
        forcing = PressureGradientForcing(
            grid=grid,
            pressure_gradient=1.0,  # unit pressure gradient in x-direction
        )

        # Apply forcing to velocity field
        force = forcing(velocity=(u, v))
    """

    def __init__(
        self,
        pressure_gradient: float = 1.0,
        force_vector: Optional[Tuple[float, ...]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            vorticity=False,  # This is a velocity forcing
            **kwargs,
        )

        self.pressure_gradient = pressure_gradient

        # Set default force vector if not provided
        if force_vector is None:
            if self.grid.ndim == 2:
                self.force_vector = (pressure_gradient, 0.0)
            else:
                raise ValueError(f"Unsupported grid dimension: {self.grid.ndim}")
        else:
            if len(force_vector) != self.grid.ndim:
                raise ValueError(
                    f"Force vector length ({len(force_vector)}) must match "
                    f"grid dimensions ({self.grid.ndim})"
                )
            self.force_vector = force_vector

    def velocity_eval(
        self,
        grid: Optional[Grid] = None,
        velocity: Optional[VectorField] = None,
        time: Optional[float] = None,
    ) -> VectorField:
        """
        Evaluate pressure gradient forcing for velocity field.

        Args:
            grid: computational grid (uses self.grid if None)
            velocity: velocity field components (used to get grid structure)

        Returns:
            GridVariableVector containing the pressure gradient forcing
        """
        grid = self.grid if grid is None else grid

        # If velocity is provided, use its structure; otherwise use default offsets
        if velocity is not None:
            # Use the same offsets and grids as the velocity components
            force_components = []
            for i, (force_magnitude, vel_component) in enumerate(
                zip(self.force_vector, velocity)
            ):
                if isinstance(vel_component, GridVariable):
                    # Create forcing with same offset and grid as velocity component
                    force_data = force_magnitude * torch.ones_like(vel_component.data)
                    force_component = GridVariable(
                        force_data, vel_component.offset, vel_component.grid
                    )
                else:
                    # Fallback: assume vel_component is a tensor
                    force_data = force_magnitude * torch.ones_like(vel_component)
                    force_component = GridVariable(
                        force_data,
                        self.offsets[i] if i < len(self.offsets) else grid.cell_center,
                        grid,
                    )
                force_components.append(force_component)
        else:
            # Use default offsets (typically cell faces for velocity)
            force_components = []
            for i, force_magnitude in enumerate(self.force_vector):
                offset = self.offsets[i] if i < len(self.offsets) else grid.cell_center

                # Create coordinate meshes to get the right shape
                coords = grid.mesh(offset)
                force_data = force_magnitude * torch.ones_like(coords[0])

                force_component = GridVariable(force_data, offset, grid)
                force_components.append(force_component)

        return GridVariableVector(tuple(force_components))
