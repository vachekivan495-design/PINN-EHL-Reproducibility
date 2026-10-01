from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import GridConfig


@dataclass(frozen=True)
class Grid:
    x: np.ndarray
    y: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    dx: float
    dy: float

    @classmethod
    def from_config(cls, config: GridConfig) -> "Grid":
        config.validate()
        x = np.linspace(config.x_min, config.x_max, config.nx, dtype=np.float64)
        y = np.linspace(config.y_min, config.y_max, config.ny, dtype=np.float64)
        X, Y = np.meshgrid(x, y, indexing="ij")
        return cls(
            x=x,
            y=y,
            X=X,
            Y=Y,
            dx=float(x[1] - x[0]),
            dy=float(y[1] - y[0]),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.X.shape

