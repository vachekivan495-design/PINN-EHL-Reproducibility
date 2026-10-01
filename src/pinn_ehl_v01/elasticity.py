from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy import fft


ELASTIC_COEFFICIENT = 0.2026423


def _influence_entry(i: int, j: int) -> float:
    def s(a: float, b: float) -> float:
        return a + math.sqrt(a * a + b * b)

    xp = i + 0.5
    xm = i - 0.5
    yp = j + 0.5
    ym = j - 0.5
    return (
        xp * math.log(s(yp, xp) / s(ym, xp))
        + ym * math.log(s(xm, ym) / s(xp, ym))
        + xm * math.log(s(ym, xm) / s(yp, xm))
        + yp * math.log(s(xp, yp) / s(xm, yp))
    )


def influence_quadrant(nx: int, ny: int) -> np.ndarray:
    kernel = np.empty((nx, ny), dtype=np.float64)
    for i in range(nx):
        for j in range(ny):
            kernel[i, j] = _influence_entry(i, j)
    return kernel


@dataclass
class ElasticConvolver:
    nx: int
    ny: int
    dx: float
    pressure_to_deformation_ratio: float
    kernel: np.ndarray
    kernel_fft: np.ndarray

    @classmethod
    def build(
        cls,
        nx: int,
        ny: int,
        dx: float,
        pressure_to_deformation_ratio: float = 1.0,
    ) -> "ElasticConvolver":
        quadrant = influence_quadrant(nx, ny)
        full = np.zeros((2 * nx - 1, 2 * ny - 1), dtype=np.float64)
        full[:nx, :ny] = quadrant
        full[:nx, ny:] = np.fliplr(quadrant[:, 1:])
        full[nx:, ny:] = np.flipud(np.fliplr(quadrant[1:, 1:]))
        full[nx:, :ny] = np.flipud(quadrant[1:, :])
        return cls(
            nx=nx,
            ny=ny,
            dx=float(dx),
            pressure_to_deformation_ratio=float(pressure_to_deformation_ratio),
            kernel=quadrant,
            kernel_fft=fft.rfft2(full),
        )

    def __call__(self, pressure: np.ndarray) -> np.ndarray:
        if pressure.shape != (self.nx, self.ny):
            raise ValueError(f"Expected pressure shape {(self.nx, self.ny)}, got {pressure.shape}")
        shape = (2 * self.nx - 1, 2 * self.ny - 1)
        padded = np.zeros(shape, dtype=np.float64)
        padded[: self.nx, : self.ny] = pressure
        convolution = fft.irfft2(fft.rfft2(padded) * self.kernel_fft, s=shape)
        return (
            convolution[: self.nx, : self.ny]
            * self.dx
            * ELASTIC_COEFFICIENT
            * self.pressure_to_deformation_ratio
        )

