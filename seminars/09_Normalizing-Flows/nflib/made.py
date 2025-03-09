"""
Implements a Masked Autoregressive MLP, where carefully constructed binary masks
over weights ensure the autoregressive property.

Source: https://github.com/karpathy/pytorch-made
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class MaskedLinear(nn.Linear):
    """
    Linear layer with a configurable mask on the weights.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))
        
    def set_mask(self, mask: np.ndarray) -> None:
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.mask * self.weight, self.bias)

class MADE(nn.Module):
    def __init__(self, nin: int, hidden_sizes: list, nout: int, num_masks: int = 1, natural_ordering: bool = False):
        """
        Parameters:
            nin: number of inputs.
            hidden_sizes: list of integers for the hidden layer sizes.
            nout: number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
            num_masks: number of mask orderings for an ensemble.
            natural_ordering: if True, use natural ordering; otherwise use random permutations.
        """
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"
        
        # Build a simple MLP
        self.net = []
        hs = [nin] + hidden_sizes + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                MaskedLinear(h0, h1),
                nn.ReLU(),
            ])
        self.net.pop()  # Remove the last ReLU
        self.net = nn.Sequential(*self.net)
        
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0
        
        self.m = {}
        self.update_masks()
        
    def update_masks(self) -> None:
        if self.m and self.num_masks == 1:
            return
        L = len(self.hidden_sizes)
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks
        
        self.m[-1] = np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin)
        for l in range(L):
            self.m[l] = rng.randint(self.m[l - 1].min(), self.nin - 1, size=self.hidden_sizes[l])
        
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])
        
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)
        
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
