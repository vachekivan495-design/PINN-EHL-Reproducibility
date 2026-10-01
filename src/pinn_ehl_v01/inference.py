"""Label-free four-condition inference. No reference field files are opened."""
from __future__ import annotations

import copy
import numpy as np
import torch

from .config import CaseConfig
from .contracts import CHECKPOINT_SCHEMA
from .data import CONDITION_FIELDS, ConditionScaler
from .model import PressureOffsetNet, TorchElasticConvolver, close_film
from .physics import HertzScaling


class Predictor:
    def __init__(self, checkpoint, device="cpu"):
        self.device = torch.device(device)
        self.checkpoint = torch.load(checkpoint, map_location=self.device, weights_only=False)
        if self.checkpoint.get("schema_version") != CHECKPOINT_SCHEMA:
            raise ValueError("Checkpoint predates the corrected closure/residual protocol")
        self.context = self.checkpoint["data_contract"]["physical_context"]
        sc = self.checkpoint["condition_scaler"]
        self.scaler = ConditionScaler(np.asarray(sc["mean"]), np.asarray(sc["scale"]))
        self.model = PressureOffsetNet(**self.checkpoint["model_config"]).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()
        grid = self.context["grid"]
        self.x = np.linspace(grid["x_min"], grid["x_max"], grid["nx"])
        self.y = np.linspace(grid["y_min"], grid["y_max"], grid["ny"])
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        self.X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        self.Y = torch.as_tensor(Y, dtype=torch.float32, device=self.device)
        self.coords = torch.stack((self.X.flatten(), self.Y.flatten()), dim=-1)
        self.convolver = TorchElasticConvolver(len(self.x), len(self.y), self.x[1] - self.x[0]).to(self.device)

    @torch.inference_mode()
    def __call__(self, conditions, cpu_output=True):
        values = np.asarray(conditions, dtype=np.float64).reshape(-1, 4)
        scalings, floors = [], []
        for index, row in enumerate(values):
            raw = copy.deepcopy(self.context)
            raw.update(dict(zip(CONDITION_FIELDS, row.tolist())))
            raw["case_id"] = f"inference_{index:06d}"
            case = CaseConfig.from_mapping(raw)
            scaling = HertzScaling.circular_point_contact(case)
            scalings.append(scaling.to_dict())
            floors.append(case.solver.film_floor_m / scaling.film_scale_m)
        condition = torch.as_tensor(self.scaler.transform(values), dtype=torch.float32, device=self.device)
        coords = self.coords.unsqueeze(0).expand(len(values), -1, -1)
        p, h0, direct = self.model(coords, condition)
        p = p.reshape(len(values), len(self.x), len(self.y))
        floor = torch.tensor(floors, dtype=p.dtype, device=self.device)
        h = close_film(p, h0, self.X, self.Y, self.convolver, floor) if self.model.film_mode == "elastic_closure" else direct.reshape_as(p) + floor[:, None, None]
        result = {"P": p, "H": h, "h0": h0, "scalings": scalings}
        if cpu_output:
            result.update({key: value.cpu().numpy() if value is not None else None for key, value in (("P", p), ("H", h), ("h0", h0))})
        return result
