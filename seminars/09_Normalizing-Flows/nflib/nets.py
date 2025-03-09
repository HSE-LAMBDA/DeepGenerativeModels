"""
Various helper network modules.
"""

import torch
from torch import nn

from nflib.made import MADE

class LeafParam(nn.Module):
    """
    Outputs a parameter tensor independent of the input.
    """
    def __init__(self, n: int):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(1, n))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.p.expand(x.size(0), self.p.size(1))

class PositionalEncoder(nn.Module):
    """
    Expands each input dimension using sine and cosine functions.
    """
    def __init__(self, freqs=(0.5, 1, 2, 4, 8)):
        super().__init__()
        self.freqs = freqs
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sines = [torch.sin(x * f) for f in self.freqs]
        coses = [torch.cos(x * f) for f in self.freqs]
        out = torch.cat(sines + coses, dim=1)
        return out

class MLP(nn.Module):
    """A simple 4-layer MLP."""
    def __init__(self, nin: int, nout: int, nh: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class PosEncMLP(nn.Module):
    """
    Position Encoded MLP, where the first layer performs position encoding.
    Each dimension of the input gets transformed to len(freqs)*2 dimensions
    using a fixed transformation of sin/cos of given frequencies.
    """
    def __init__(self, nin: int, nout: int, nh: int, freqs=(0.5, 1, 2, 4, 8)):
        super().__init__()
        self.net = nn.Sequential(
            PositionalEncoder(freqs),
            MLP(nin * len(freqs) * 2, nout, nh),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ARMLP(nn.Module):
    """
    Auto-regressive MLP, implemented using the MADE network.
    """
    def __init__(self, nin: int, nout: int, nh: int):
        super().__init__()
        self.net = MADE(nin, [nh, nh, nh], nout, num_masks=1, natural_ordering=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
