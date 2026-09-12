import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from makani.models.common import SpectralConv
from makani.models.common.factorizations import get_contract_fun

class AdaptableSpectralConv(nn.Module):
    """Base class for wrapping SpectralConv that adapts Fourier weights for coarsening 
    This is useful for multi-grid FNOs where the same operator is applied at different resolutions.
    
    Once trained, network weights can be precomputed for any desired coarsening level.
    """

    def __init__(self, spectral_conv: SpectralConv):
        super().__init__()
        self.spectral_conv = spectral_conv
        self.operator_type = self.spectral_conv.operator_type or "dhconv"
        if self.operator_type == "diagonal":
            self.total_modes = self.spectral_conv.modes_lat_local * self.spectral_conv.modes_lon_local
        else:
            self.total_modes = self.spectral_conv.modes_lat_local
        self.hidden_dim = 2 * self.total_modes
        self.weight_transform = self.default_weight_transform
        self.base_coarsening_level = 0

    def default_weight_transform(self, W_flat: torch.Tensor):
        raise NotImplementedError("Subclasses must define a weight transform method.")
    
    def _flatten_weight(self, W0: torch.Tensor) -> torch.Tensor:
        """Reshape weight tensor to (..., total_modes) for weight transform input."""
        dims = W0.ndim 
        if self.operator_type == "diagonal":
            return W0.reshape(*W0.shape[:-2], -1)
        return W0

    def coarsen_transform(self, W_real, W_imag):
            return (self.weight_transform(W_real) + W_real,
                    self.weight_transform(W_imag) + W_imag)
    
    def _apply_weight_transform(self, W_flat: torch.Tensor,
                                         coarsening_level: int) -> torch.Tensor:
        """`AdaptableSpectralConv._apply_weight_transform` with each application checkpointed.

        Real and imaginary parts stay unpacked across the loop instead of being recombined into a
        complex tensor and split again each time — `torch.complex(a, b).real` returns exactly `a`, so
        this is bit-identical, and it keeps one checkpointed segment per application rather than two.
        """
       

        W_real, W_imag = W_flat.real, W_flat.imag
        for _ in range(coarsening_level):
            if torch.is_grad_enabled():
                W_real, W_imag = checkpoint(self.coarsen_transform, W_real, W_imag, use_reentrant=False)
            else:
                # No backward to save for — this is the path validation and merge_weights take.
                W_real, W_imag = self.coarsen_transform(W_real, W_imag)
        return torch.complex(W_real, W_imag)
    
    def forward(self, x: torch.Tensor, coarsening_level: int = 0, output_shape=None):
        if coarsening_level == 0:
            return self.spectral_conv.forward(x)

        W0 = self.spectral_conv.weight
        if not torch.is_tensor(W0):
            W0 = W0.to_tensor()

        W_flat = self._flatten_weight(W0)
        W_adapted = self._apply_weight_transform(W_flat, coarsening_level).view_as(W0)

        return self.spectral_conv.forward(x, override_weight=W_adapted)

    def merge_weights(self, coarsening_level: int):
        """Merge the MLP-adapted weights into the SpectralConv weight tensor for a given coarsening level.
        This allows for precomputing the adapted weights and removing the MLP from the forward pass.
        """
        if coarsening_level == 0:
            return

        W0 = self.spectral_conv.weight
        if not torch.is_tensor(W0):
            W0 = W0.to_tensor()

        W_flat = self._flatten_weight(W0)
        W_adapted = self._apply_weight_transform(W_flat, coarsening_level).view_as(W0)

        self.spectral_conv.weight = nn.Parameter(W_adapted.detach())
        self.base_coarsening_level += coarsening_level


def coarsen_model(model: nn.Module, coarsening_level: int) -> nn.Module:
    """Merge MLP-adapted weights into SpectralConv weights in a model to coarsen it.

    Recursively walks the module tree and calls merge_mlp_weights(coarsening_level) on each
    AdaptableSpectralConv layer in-place. This effectively updates the underlying
    SpectralConv weights to reflect the coarsened representation.

    Works with nested modules including ``nn.ModuleList``.

    Parameters
    ----------
    model : nn.Module
        Model whose AdaptableSpectralConv layers should be merged into SpectralConv.
    coarsening_level : int
        Number of levels of coarsening to apply to each AdaptableSpectralConv-wrapped SpectralConv layer.

    Returns
    -------
    nn.Module
        The same model object with coarsened SpectralConv weights.
    """
    for _, module in model.named_children():
        if isinstance(module, AdaptableSpectralConv):
            module.merge_weights(coarsening_level)
        else:
            coarsen_model(module, coarsening_level)
    return model

class MLPAdaptableSpectralConv(AdaptableSpectralConv):
    """Wrapper for SpectralConv that adapts weights for coarsening by applying an MLP to the Fourier weights.
    This is useful for multi-grid FNOs where the same operator is applied at different resolutions.
    The MLP is applied independently to the real and imaginary parts of the complex weights.
    Note that the same MLP is applied to all channels; only mode weights are affected by coarsening.

    Once trained, network weights can be precomputed for any desired coarsening level.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_transform = nn.Sequential(
            nn.Linear(self.total_modes, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.total_modes),
        )


class SwiGLU(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W1 = nn.Linear(d_in, d_out)
        self.W2 = nn.Linear(d_in, d_out)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.W1(x) * self.silu(self.W2(x))

class GLUAdaptableSpectralConv(AdaptableSpectralConv):
    """Wrapper for SpectralConv that adapts weights for coarsening by applying a Gated Linear Unit (GLU)-based 
    transformation to the Fourier weights.
    This is useful for multi-grid FNOs where the same operator is applied at different resolutions.
    The GLU is applied independently to the real and imaginary parts of the complex weights.
    Note that the same GLU is applied to all channels; only mode weights are affected by coarsening.

    Once trained, network weights can be precomputed for any desired coarsening level.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_transform = nn.Sequential(
            nn.Linear(self.total_modes, self.hidden_dim),
            nn.GELU(),
            SwiGLU(self.hidden_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.total_modes),
        )
