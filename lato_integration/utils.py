"""
Foundation utilities adapted from LATO (ICML 2026).

Provides DiagonalGaussianDistribution for proper VAE posterior handling
with clamped logvar, KL divergence, and NLL computation.
"""

import torch
import numpy as np
from typing import Union, List


class DiagonalGaussianDistribution(object):
    """
    A diagonal Gaussian distribution parameterized by mean and log-variance.

    Adapted from LATO: lato/modules/utils.py

    **FP16 SAFE**: All internal distribution computations (exp, kl, nll) are
    performed in float32 regardless of input dtype, preventing overflow from
    exp(logvar) when logvar > 11 (FP16 max ≈ 65504 = exp(11.09)).

    Args:
        parameters: Either a tensor that will be chunked into mean and logvar,
                    or a list/tuple of [mean, logvar].
        deterministic: If True, variance is set to zero.
        feat_dim: The dimension along which to chunk parameters.
    """

    def __init__(
        self,
        parameters: Union[torch.Tensor, List[torch.Tensor]],
        deterministic: bool = False,
        feat_dim: int = 1,
    ):
        self.feat_dim = feat_dim
        self.parameters = parameters

        if isinstance(parameters, list):
            self.mean = parameters[0]
            self.logvar = parameters[1]
        else:
            self.mean, self.logvar = torch.chunk(parameters, 2, dim=feat_dim)

        # Record original dtype for sample/mode returns
        self._input_dtype = self.mean.dtype

        # ---- FP16 SAFETY: upcast to float32 for all internal computations ----
        # FP16 max ≈ 65504, so exp(x) overflows for x > 11.09.
        # With logvar clamped to 20, exp(20) = 4.85e8 → inf in FP16.
        # We always compute in float32 and cast results back as needed.
        if self.mean.dtype in (torch.float16, torch.bfloat16):
            self.mean = self.mean.float()
            self.logvar = self.logvar.float()

        # Clamp logvar for numerical stability
        # Upper bound tightened from 20→10 for FP16 safety (exp(10)≈22026 < 65504)
        self.logvar = torch.clamp(self.logvar, -30.0, 10.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean)

    def sample(self) -> torch.Tensor:
        """Sample from the posterior using reparameterization."""
        x = self.mean + self.std * torch.randn_like(self.mean)
        # Cast back to original dtype for downstream FP16 model compatibility
        if self._input_dtype != x.dtype:
            x = x.to(self._input_dtype)
        return x

    def kl(self, other=None, dims=(1, 2, 3)) -> torch.Tensor:
        """
        Compute KL divergence. Always computed in float32.

        If other is None, computes KL(q || N(0, I)).
        Otherwise, computes KL(q || other).
        """
        if self.deterministic:
            return torch.Tensor([0.0]).to(self.mean.device)

        if other is None:
            return 0.5 * torch.mean(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                dim=dims,
            )
        else:
            return 0.5 * torch.mean(
                torch.pow(self.mean - other.mean, 2) / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar,
                dim=dims,
            )

    def nll(self, sample, dims=(1, 2, 3)) -> torch.Tensor:
        """Compute negative log-likelihood. Always computed in float32."""
        if self.deterministic:
            return torch.Tensor([0.0]).to(self.mean.device)
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        """Return the mode of the distribution (the mean)."""
        x = self.mean
        if self._input_dtype != x.dtype:
            x = x.to(self._input_dtype)
        return x
