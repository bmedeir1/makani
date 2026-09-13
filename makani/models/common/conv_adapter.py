# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from makani.models.common.spectral_convolution import SpectralConv
from makani.utils import comm

class AdaptableSpectralConv(nn.Module):
    """Base class for wrapping SpectralConv that adapts Fourier weights for coarsening 
    This is useful for multi-grid FNOs where the same operator is applied at different resolutions.
    
    Once trained, network weights can be precomputed for any desired coarsening level.
    """

    def __init__(self, spectral_conv: SpectralConv):
        super().__init__()
        self.spectral_conv = spectral_conv
        self.operator_type = self.spectral_conv.operator_type

        self._check_modes_not_sharded()

        if self.operator_type == "diagonal":
            self.total_modes = self.spectral_conv.modes_lat_local * self.spectral_conv.modes_lon_local
        else:
            self.total_modes = self.spectral_conv.modes_lat_local
        self.hidden_dim = 2 * self.total_modes
        self.weight_transform = self.default_weight_transform
        self.base_coarsening_level = 0

    def _check_modes_not_sharded(self):
        """Reject any model-parallel layout that splits the mode axes the transform mixes over.

        The transform is a dense map over the spectral modes, so it needs the whole mode axis on
        one rank. ``SpectralConv`` itself is pointwise in ``l``, which is why it can afford to
        shard that axis over the "h" group (see ``SpectralConv.__init__``: ``modes_lat_local`` is
        ``l_shapes[rank("h")]`` and the weight is tagged ``sharded_dims_mp[-1] = "h"``). Applying
        the transform to a slice of ``l`` is not the restriction of applying it to all of ``l``,
        so a sharded run would silently compute a different operator than a single-rank run
        rather than fail. Supporting it needs a gather over the mode axis around the transform.
        """
        sharded = []
        if comm.get_size("h") > 1:
            sharded.append(f"'h' splits the l axis across {comm.get_size('h')} ranks")
        if self.operator_type == "diagonal" and comm.get_size("w") > 1:
            sharded.append(f"'w' splits the m axis across {comm.get_size('w')} ranks")

        if sharded:
            raise NotImplementedError(
                f"{type(self).__name__} mixes across the spectral modes and requires them "
                f"unsharded, but {' and '.join(sharded)}. Run with h=1 "
                f"{'(and w=1 for operator_type=diagonal) ' if self.operator_type == 'diagonal' else ''}"
                f"or add a gather over the mode axis in _apply_weight_transform."
            )

    def _tag_transform_parameters(self):
        """Annotate the transform's parameters as replicated across all model-parallel groups.

        ``_check_modes_not_sharded`` has already established that no mode axis is split, so every
        model rank builds an identical transform. ``is_shared_mp`` is iterated as a list of comm
        group names and ``sharded_dims_mp`` is indexed per dimension (see
        ``mpu/mappings.py:init_gradient_reduction_hooks`` and ``convert_checkpoint.py``), so
        "replicated, no sharded dims" is spelled this way rather than with None or False.
        Subclasses call this once they have built ``weight_transform``.
        """
        for param in self.weight_transform.parameters():
            param.is_shared_mp = ["model"]
            param.sharded_dims_mp = [None] * param.ndim

    def default_weight_transform(self, W_flat: torch.Tensor):
        raise NotImplementedError("Subclasses must define a weight transform method.")

    def _flatten_weight(self, W0: torch.Tensor) -> torch.Tensor:
        """Reshape weight tensor to (..., modes) for weight transform input."""
        if self.operator_type == "diagonal":
            return W0.reshape(*W0.shape[:-2], -1)
        return W0

    def coarsen_transform(self, W_real, W_imag):
        return (self.weight_transform(W_real) + W_real,
                self.weight_transform(W_imag) + W_imag)


    def _apply_weight_transform(self, W_flat: torch.Tensor,
                                         coarsening_level: int) -> torch.Tensor:
        """Apply ``weight_transform`` with a skip connection ``coarsening_level`` times.

        Operates on real-valued tensors, so the real and imaginary parts of the complex weight are
        adapted independently with shared parameters. Residual formulation to match LoRA-style
        adaptation, where the original weight is added back to the transformed weight.

        Each application is gradient-checkpointed: the saved activations are the Fourier weights
        themselves, so at high coarsening levels keeping every application's activations dominates
        memory. Checkpointing leaves one application in flight and recomputes it in the backward;
        the gradient is exact, not truncated.

        Real and imaginary parts stay unpacked across the loop instead of being recombined into a
        complex tensor and split again each time — `torch.complex(a, b).real` returns exactly `a`, so
        this is bit-identical, and it keeps one checkpointed segment per application rather than two.
        """
        W_real, W_imag = W_flat.real, W_flat.imag
        for _ in range(coarsening_level):
            if torch.is_grad_enabled():
                # use_reentrant=False is required: when the base model is frozen
                # the weights entering the first application have requires_grad=False and only the
                # transform's own parameters need gradients. The reentrant implementation reads
                # that as nothing to differentiate and silently returns an output with no grad_fn.
                W_real, W_imag = checkpoint(self.coarsen_transform, W_real, W_imag, use_reentrant=False)
            else:
                # No backward to save for — this is the path validation and merge_weights take.
                W_real, W_imag = self.coarsen_transform(W_real, W_imag)
        return torch.complex(W_real, W_imag)
    
    def forward(self, x: torch.Tensor, coarsening_level: int = 0):
        """Apply the wrapped convolution with weights adapted ``coarsening_level`` times.

        ``coarsening_level`` is applied on top of whatever the underlying weights already
        represent, so after ``merge_weights(k)`` a call with ``coarsening_level=j`` yields level
        ``k + j``, not level ``j``. ``base_coarsening_level`` records the merged part.
        """
        if coarsening_level == 0:
            return self.spectral_conv.forward(x)

        W0 = self.spectral_conv.weight
        if not torch.is_tensor(W0):
            W0 = W0.to_tensor()

        W_flat = self._flatten_weight(W0)
        W_adapted = self._apply_weight_transform(W_flat, coarsening_level).view_as(W0)

        return self.spectral_conv.forward(x, override_weight=W_adapted)

    def merge_weights(self, coarsening_level: int):
        """Merge the adapted weights into the SpectralConv weight tensor for a given coarsening level.
        This allows for precomputing the adapted weights and removing the MLP from the forward pass.
        """
        if coarsening_level == 0:
            return

        W0 = self.spectral_conv.weight
        if not torch.is_tensor(W0):
            W0 = W0.to_tensor()

        W_flat = self._flatten_weight(W0)
        W_adapted = self._apply_weight_transform(W_flat, coarsening_level).view_as(W0)

        old_weight = self.spectral_conv.weight
        self.spectral_conv.weight = nn.Parameter(W_adapted.detach())
        # carry over the model-parallel annotations SpectralConv.__init__ attached to the weight;
        # a fresh nn.Parameter would otherwise drop them
        for attr in ("is_shared_mp", "sharded_dims_mp"):
            if hasattr(old_weight, attr):
                setattr(self.spectral_conv.weight, attr, getattr(old_weight, attr))

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
        self._tag_transform_parameters()


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
        self._tag_transform_parameters()
