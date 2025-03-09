"""
Implements various flows.
Each flow is invertible so it can be forward()ed and inverse()ed.
Each flow also outputs its log det J "regularization"

Reference:

NICE: Non-linear Independent Components Estimation, Dinh et al. 2014
https://arxiv.org/abs/1410.8516

Variational Inference with Normalizing Flows, Rezende and Mohamed 2015
https://arxiv.org/abs/1505.05770

Density estimation using Real NVP, Dinh et al. May 2016
https://arxiv.org/abs/1605.08803
(Laurent's extension of NICE)

Improved Variational Inference with Inverse Autoregressive Flow, Kingma et al June 2016
https://arxiv.org/abs/1606.04934
(IAF)

Masked Autoregressive Flow for Density Estimation, Papamakarios et al. May 2017 
https://arxiv.org/abs/1705.07057
"The advantage of Real NVP compared to MAF and IAF is that it can both generate data and estimate densities with one forward pass only, whereas MAF would need D passes to generate data and IAF would need D passes to estimate densities."
(MAF)

Glow: Generative Flow with Invertible 1x1 Convolutions, Kingma and Dhariwal, Jul 2018
https://arxiv.org/abs/1807.03039

"Normalizing Flows for Probabilistic Modeling and Inference"
https://arxiv.org/abs/1912.02762
(review paper)
"""

import torch
from torch import nn

from nflib.nets import LeafParam, MLP, ARMLP

class AffineConstantFlow(nn.Module):
    """
    Scales + Shifts the flow by (learned) constants per dimension.
    In the NICE paper, the scaling layer is a special case of this when t is None.
    """
    def __init__(self, dim: int, scale: bool = True, shift: bool = True):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, dim)) if scale else None
        self.t = nn.Parameter(torch.randn(1, dim)) if shift else None
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.s if self.s is not None else torch.zeros_like(x)
        t = self.t if self.t is not None else torch.zeros_like(x)
        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.s if self.s is not None else torch.zeros_like(z)
        t = self.t if self.t is not None else torch.zeros_like(z)
        x = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=1)
        return x, log_det

class ActNorm(AffineConstantFlow):
    """
    Affine constant flow with data-dependent initialization, as described in the Glow paper.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = False
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Use the first batch to initialize s and t so that the output is unit Gaussian.
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None, "ActNorm requires both scale and shift."
            self.s.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
            self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
            self.data_dep_init_done = True
        return super().forward(x)

class AffineHalfFlow(nn.Module):
    """
    Affine autoregressive flow: half of the dimensions in x are transformed as a function of the other half.
    """
    def __init__(self, dim: int, parity: bool, net_class=MLP, nh: int = 24, scale: bool = True, shift: bool = True):
        super().__init__()
        self.dim = dim
        self.parity = parity
        self.s_cond = lambda x: torch.zeros(x.size(0), self.dim // 2, device=x.device, dtype=x.dtype)
        self.t_cond = lambda x: torch.zeros(x.size(0), self.dim // 2, device=x.device, dtype=x.dtype)
        if scale:
            self.s_cond = net_class(self.dim // 2, self.dim // 2, nh)
        if shift:
            self.t_cond = net_class(self.dim // 2, self.dim // 2, nh)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = x[:, ::2], x[:, 1::2]
        if self.parity:
            x0, x1 = x1, x0
        s = self.s_cond(x0)
        t = self.t_cond(x0)
        z0 = x0
        z1 = torch.exp(s) * x1 + t
        if self.parity:
            z0, z1 = z1, z0
        z = torch.cat([z0, z1], dim=1)
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z0, z1 = z[:, ::2], z[:, 1::2]
        if self.parity:
            z0, z1 = z1, z0
        s = self.s_cond(z0)
        t = self.t_cond(z0)
        x0 = z0
        x1 = (z1 - t) * torch.exp(-s)
        if self.parity:
            x0, x1 = x1, x0
        x = torch.cat([x0, x1], dim=1)
        log_det = torch.sum(-s, dim=1)
        return x, log_det

class SlowMAF(nn.Module):
    """
    Masked Autoregressive Flow (slow version) with an explicit network per input dimension.
    """
    def __init__(self, dim: int, parity: bool, net_class=MLP, nh: int = 24):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleDict()
        self.layers["0"] = LeafParam(2)
        for i in range(1, dim):
            self.layers[str(i)] = net_class(i, 2, nh)
        self.order = list(range(dim)) if parity else list(range(dim))[::-1]
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros_like(x)
        log_det = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        for i in range(self.dim):
            st = self.layers[str(i)](x[:, :i])
            s, t = st[:, 0], st[:, 1]
            z[:, self.order[i]] = x[:, i] * torch.exp(s) + t
            log_det += s
        return z, log_det

    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        for i in range(self.dim):
            st = self.layers[str(i)](x[:, :i])
            s, t = st[:, 0], st[:, 1]
            x[:, i] = (z[:, self.order[i]] - t) * torch.exp(-s)
            log_det += -s
        return x, log_det

class MAF(nn.Module):
    """
    Masked Autoregressive Flow using a MADE-style network for fast parallel density estimation.
    """
    def __init__(self, dim: int, parity: bool, net_class=ARMLP, nh: int = 24):
        super().__init__()
        self.dim = dim
        self.net = net_class(dim, dim * 2, nh)
        self.parity = parity

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        st = self.net(x)
        s, t = st.split(self.dim, dim=1)
        z = x * torch.exp(s) + t
        z = z.flip(dims=(1,)) if self.parity else z
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        z = z.flip(dims=(1,)) if self.parity else z
        for i in range(self.dim):
            st = self.net(x.clone())
            s, t = st.split(self.dim, dim=1)
            x[:, i] = (z[:, i] - t[:, i]) * torch.exp(-s[:, i])
            log_det += -s[:, i]
        return x, log_det

class IAF(MAF):
    """
    Inverse Autoregressive Flow (IAF): swaps forward and inverse so that sampling is fast.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Swap the roles of forward and inverse
        self.forward, self.inverse = self.inverse, self.forward

class Invertible1x1Conv(nn.Module):
    """
    Invertible 1x1 Convolution layer as introduced in the Glow paper.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))
        # Use new LU factorization API
        LU, pivots = torch.linalg.lu_factor(Q)
        P, L, U = torch.lu_unpack(LU, pivots)
        self.register_buffer("P", P)
        self.L = nn.Parameter(L)
        self.S = nn.Parameter(U.diag())
        self.U = nn.Parameter(torch.triu(U, diagonal=1))

    def _assemble_W(self) -> torch.Tensor:
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim, device=self.L.device, dtype=self.L.dtype))
        U = torch.triu(self.U, diagonal=1)
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        W = self._assemble_W()
        z = x @ W
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return z, log_det

    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        W = self._assemble_W()
        W_inv = torch.linalg.inv(W)
        x = z @ W_inv
        log_det = -torch.sum(torch.log(torch.abs(self.S)))
        return x, log_det

class NormalizingFlow(nn.Module):
    """
    A sequence of normalizing flows.
    """
    def __init__(self, flows):
        super().__init__()
        self.flows = nn.ModuleList(flows)

    def forward(self, x: torch.Tensor) -> tuple[list, torch.Tensor]:
        m, _ = x.shape
        log_det = torch.zeros(m, device=x.device, dtype=x.dtype)
        zs = [x]
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def inverse(self, z: torch.Tensor) -> tuple[list, torch.Tensor]:
        m, _ = z.shape
        log_det = torch.zeros(m, device=z.device, dtype=z.dtype)
        xs = [z]
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z)
            log_det += ld
            xs.append(z)
        return xs, log_det

class NormalizingFlowModel(nn.Module):
    """
    A Normalizing Flow Model is a (prior, flow) pair.
    """
    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows)
    
    def forward(self, x: torch.Tensor) -> tuple[list, torch.Tensor, torch.Tensor]:
        zs, log_det = self.flow.forward(x)
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def inverse(self, z: torch.Tensor) -> tuple[list, torch.Tensor]:
        xs, log_det = self.flow.inverse(z)
        return xs, log_det
    
    def sample(self, num_samples: int) -> list:
        z = self.prior.sample((num_samples,))
        xs, _ = self.flow.inverse(z)
        return xs
