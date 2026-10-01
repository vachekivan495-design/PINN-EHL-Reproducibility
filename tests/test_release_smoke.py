from pathlib import Path

import numpy as np
import torch

from pinn_ehl_v01.inference import Predictor


ROOT = Path(__file__).resolve().parents[1]


def test_checkpoint_architecture_and_hard_constraints() -> None:
    predictor = Predictor(ROOT / "artifacts" / "primary_1pct_seed1_inference.pt")
    assert predictor.checkpoint["model_config"]["depth"] == 6
    assert predictor.checkpoint["model_config"]["fourier_features"] == 64
    output = predictor([[40.0, 1.5, 0.07, 2.0e-8]])
    pressure = output["P"][0]
    assert pressure.shape == (769, 769)
    assert np.all(pressure >= 0.0)
    assert float(np.max(pressure[[0, -1], :])) == 0.0
    assert float(np.max(pressure[:, [0, -1]])) == 0.0
    np.testing.assert_allclose(pressure, pressure[:, ::-1], rtol=0.0, atol=2e-6)
    assert torch.isfinite(torch.as_tensor(output["H"])).all()

