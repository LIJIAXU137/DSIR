# -*- coding: utf-8 -*-

import torch
from .ACIL import ACILLearner, ACIL
from .AnalyticLinear import GeneralizedDSIR

__all__ = ["DSIR", "DSIRLearner", "GeneralizedDSIRLearner"]


class DSIR(ACIL):
    def fit(self, X: torch.Tensor, y: torch.Tensor, *args, **kwargs) -> None:
        X = self.feature_expansion(X)
        self.analytic_linear.fit(X, y)


class DSIRLearner(ACILLearner):
    def make_model(self) -> None:
        self.model = DSIR(
            self.backbone_output,
            self.wrap_data_parallel(self.backbone),
            self.buffer_size,
            self.gamma,
            device=self.device,
            dtype=torch.double,
            linear=GeneralizedDSIR,
        )


class GeneralizedDSIRLearner(DSIRLearner):
    pass
