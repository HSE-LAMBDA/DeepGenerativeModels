"""
Implements various normalizing flows. Each flow
can perform a forward transformation (data → latent) and an inverse transformation 
(latent → data) along with computing the log-determinant of the Jacobian (useful for 
regularization in density estimation).

Reference Papers:
- NICE: Non-linear Independent Components Estimation, Dinh et al. 2014
  https://arxiv.org/abs/1410.8516
- Variational Inference with Normalizing Flows, Rezende and Mohamed 2015
  https://arxiv.org/abs/1505.05770
- Density estimation using Real NVP, Dinh et al. May 2016
  https://arxiv.org/abs/1605.08803  (Laurent's extension of NICE)
- Improved Variational Inference with Inverse Autoregressive Flow, Kingma et al. June 2016
  https://arxiv.org/abs/1606.04934  (IAF)
- Masked Autoregressive Flow for Density Estimation, Papamakarios et al. May 2017 
  https://arxiv.org/abs/1705.07057  (MAF)
- Glow: Generative Flow with Invertible 1x1 Convolutions, Kingma and Dhariwal, July 2018
  https://arxiv.org/abs/1807.03039
- "Normalizing Flows for Probabilistic Modeling and Inference"
  https://arxiv.org/abs/1912.02762 (review paper)
"""

import torch
from torch import nn

# Importing necessary network modules from nflib
from nflib.nets import LeafParam, MLP, ARMLP

class AffineConstantFlow(nn.Module):
    """
    Flow that applies a learned per-dimension scaling and shifting.
    This is a generalization of the scaling layer from NICE, where the shift can be omitted.
    
    Attributes:
        s (nn.Parameter): Learnable scaling parameters.
        t (nn.Parameter): Learnable shifting parameters.
    """
    def __init__(self, dim: int, scale: bool = True, shift: bool = True):
        super().__init__()
        # Initialize scaling parameter if enabled; otherwise, set to None.
        self.s = nn.Parameter(torch.randn(1, dim)) if scale else None
        # Initialize shifting parameter if enabled; otherwise, set to None.
        self.t = nn.Parameter(torch.randn(1, dim)) if shift else None
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computes: z = x * exp(s) + t, with log_det = sum(s)
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            tuple: Transformed tensor and the log-determinant of the Jacobian.
        """
        # Use provided parameters or default to zeros if not enabled.
        s = self.s if self.s is not None else torch.zeros_like(x)
        t = self.t if self.t is not None else torch.zeros_like(x)
        # Apply the affine transformation.
        z = x * torch.exp(s) + t
        # Compute log-determinant: sum over dimensions.
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transformation: x = (z - t) * exp(-s), with log_det = -sum(s)
        
        Args:
            z (torch.Tensor): Transformed tensor.
        
        Returns:
            tuple: Inverted tensor and the log-determinant of the inverse transformation.
        """
        s = self.s if self.s is not None else torch.zeros_like(z)
        t = self.t if self.t is not None else torch.zeros_like(z)
        # Reverse the affine transformation.
        x = (z - t) * torch.exp(-s)
        # Log-determinant for the inverse transformation.
        log_det = torch.sum(-s, dim=1)
        return x, log_det

class ActNorm(AffineConstantFlow):
    """
    ActNorm layer with data-dependent initialization.
    Initializes scale and shift parameters based on the first batch so that
    the output activations have unit variance.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Flag to check if data-dependent initialization is completed.
        self.data_dep_init_done = False
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        On the first forward pass, initialize s and t such that the output is close to a unit Gaussian.
        Subsequent passes use the learned parameters.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            tuple: Transformed tensor and log-determinant.
        """
        if not self.data_dep_init_done:
            # Ensure both scale and shift are available for initialization.
            assert self.s is not None and self.t is not None, "ActNorm requires both scale and shift."
            # Initialize scaling parameter based on the inverse of the standard deviation.
            self.s.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
            # Initialize shifting parameter so that the mean becomes zero after scaling.
            self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
            self.data_dep_init_done = True
        # Call the parent's forward method.
        return super().forward(x)

class AffineHalfFlow(nn.Module):
    """
    Affine autoregressive flow that transforms a subset of dimensions conditioned on the remaining ones.
    Splits the input tensor into two parts: one for conditioning and one to be transformed.
    
    Attributes:
        n1 (int): Number of dimensions in the conditioning part.
        n2 (int): Number of dimensions in the transformed part.
        parity (bool): Determines the split order.
        s_cond (nn.Module or function): Network to compute scale factors.
        t_cond (nn.Module or function): Network to compute shift factors.
    """
    def __init__(self, dim: int, parity: bool, net_class=MLP, nh: int = 24, scale: bool = True, shift: bool = True):
        super().__init__()
        self.dim = dim
        self.parity = parity
        # Split input dimensions into two parts; if odd, the second part is larger.
        self.n1 = dim // 2
        self.n2 = dim - self.n1
        # Initialize the network to compute scaling if enabled; else return zeros.
        if scale:
            self.s_cond = net_class(self.n1, self.n2, nh)
        else:
            self.s_cond = lambda x: torch.zeros(x.size(0), self.n2, device=x.device, dtype=x.dtype)
        # Initialize the network to compute shifting if enabled; else return zeros.
        if shift:
            self.t_cond = net_class(self.n1, self.n2, nh)
        else:
            self.t_cond = lambda x: torch.zeros(x.size(0), self.n2, device=x.device, dtype=x.dtype)
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the affine transformation on one part of x conditioned on the other.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            tuple: Transformed tensor and log-determinant.
        """
        if not self.parity:
            # Standard order: first half conditions the second half.
            x_a = x[:, :self.n1]
            x_b = x[:, self.n1:]
        else:
            # Swap roles: latter part conditions the first part.
            x_a = x[:, self.n2:]
            x_b = x[:, :self.n2]
        # Compute scale and shift using the conditioning network.
        s = self.s_cond(x_a)
        t = self.t_cond(x_a)
        # Apply the affine transformation to the target half.
        z_b = x_b * torch.exp(s) + t
        # Reassemble the output to preserve the original order.
        if not self.parity:
            z = torch.cat([x_a, z_b], dim=1)
        else:
            z = torch.cat([z_b, x_a], dim=1)
        # Sum over the scale parameters to get the log-determinant.
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse of the affine transformation applied in the forward pass.
        
        Args:
            z (torch.Tensor): Transformed tensor.
        
        Returns:
            tuple: Reconstructed input tensor and log-determinant.
        """
        if not self.parity:
            z_a = z[:, :self.n1]
            z_b = z[:, self.n1:]
        else:
            z_a = z[:, self.n2:]
            z_b = z[:, :self.n2]
        # Recompute the scale and shift parameters from the conditioning part.
        s = self.s_cond(z_a)
        t = self.t_cond(z_a)
        # Invert the affine transformation.
        x_b = (z_b - t) * torch.exp(-s)
        if not self.parity:
            x = torch.cat([z_a, x_b], dim=1)
        else:
            x = torch.cat([x_b, z_a], dim=1)
        # Inverse log-determinant is the negative sum of scales.
        log_det = torch.sum(-s, dim=1)
        return x, log_det

class SlowMAF(nn.Module):
    """
    A slow implementation of Masked Autoregressive Flow (MAF) where each input dimension
    has its own dedicated network for transformation.
    
    Attributes:
        layers (ModuleDict): Contains networks for each input dimension.
        order (list): Ordering of dimensions for autoregressive processing.
    """
    def __init__(self, dim: int, parity: bool, net_class=MLP, nh: int = 24):
        super().__init__()
        self.dim = dim
        # Create a ModuleDict to store transformation networks for each dimension.
        self.layers = nn.ModuleDict()
        # The first dimension uses a simple parameter (leaf parameter) since it is unconditional.
        self.layers["0"] = LeafParam(2)
        # For subsequent dimensions, create a network that takes previous dimensions as input.
        for i in range(1, dim):
            self.layers[str(i)] = net_class(i, 2, nh)
        # Define processing order based on the parity flag.
        self.order = list(range(dim)) if parity else list(range(dim))[::-1]
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass processes each dimension sequentially in an autoregressive manner.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            tuple: Transformed tensor and cumulative log-determinant.
        """
        # Initialize output tensor and log-determinant.
        z = torch.zeros_like(x)
        log_det = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        # Process each dimension sequentially.
        for i in range(self.dim):
            # Obtain scale and shift parameters based on already processed dimensions.
            st = self.layers[str(i)](x[:, :i])
            s, t = st[:, 0], st[:, 1]
            # Transform the current dimension using the computed parameters.
            z[:, self.order[i]] = x[:, i] * torch.exp(s) + t
            # Accumulate the log-determinant.
            log_det += s
        return z, log_det

    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse pass reconstructs the input sequentially.
        
        Args:
            z (torch.Tensor): Transformed tensor.
        
        Returns:
            tuple: Reconstructed input tensor and cumulative log-determinant.
        """
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        # Sequentially invert each transformation.
        for i in range(self.dim):
            st = self.layers[str(i)](x[:, :i])
            s, t = st[:, 0], st[:, 1]
            # Invert the transformation for the current dimension.
            x[:, i] = (z[:, self.order[i]] - t) * torch.exp(-s)
            log_det += -s
        return x, log_det

class MAF(nn.Module):
    """
    Masked Autoregressive Flow (MAF) using a MADE-style network for fast, parallel density estimation.
    
    Attributes:
        net (nn.Module): A network that outputs both scale and shift for all dimensions.
        parity (bool): If True, the order of dimensions is reversed.
    """
    def __init__(self, dim: int, parity: bool, net_class=ARMLP, nh: int = 24):
        super().__init__()
        self.dim = dim
        # Initialize a MADE-style network that outputs 2 * dim parameters.
        self.net = net_class(dim, dim * 2, nh)
        self.parity = parity

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the forward transformation: z = x * exp(s) + t, using parallel computation.
        Optionally reverses the dimension order based on parity.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            tuple: Transformed tensor and the log-determinant.
        """
        # Compute scale and shift for all dimensions.
        st = self.net(x)
        s, t = st.split(self.dim, dim=1)
        # Apply element-wise affine transformation.
        z = x * torch.exp(s) + t
        # Optionally flip the order of dimensions.
        z = z.flip(dims=(1,)) if self.parity else z
        # Sum log-determinants from the scale factors.
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transformation: reconstructs x sequentially.
        Due to the autoregressive structure, the inverse pass must be computed sequentially.
        
        Args:
            z (torch.Tensor): Transformed tensor.
        
        Returns:
            tuple: Reconstructed input tensor and cumulative log-determinant.
        """
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        # Reverse dimension order if parity is set.
        z = z.flip(dims=(1,)) if self.parity else z
        # Process each dimension sequentially.
        for i in range(self.dim):
            # Use a clone of x for stability; compute parameters for current state.
            st = self.net(x.clone())
            s, t = st.split(self.dim, dim=1)
            # Invert the affine transformation for the i-th dimension.
            x[:, i] = (z[:, i] - t[:, i]) * torch.exp(-s[:, i])
            log_det += -s[:, i]
        return x, log_det

class IAF(MAF):
    """
    Inverse Autoregressive Flow (IAF) is a variant of MAF where the roles of forward and inverse are swapped.
    This design makes sampling efficient.
    
    Note:
        The forward method now corresponds to the inverse of the original MAF and vice versa.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Swap the forward and inverse methods for efficient sampling.
        self.forward, self.inverse = self.inverse, self.forward

class Invertible1x1Conv(nn.Module):
    """
    Invertible 1x1 Convolution layer as introduced in the Glow paper.
    Uses an LU decomposition to parameterize the weight matrix efficiently and to compute its determinant.
    
    Attributes:
        P (torch.Tensor): Permutation matrix (buffer, not learnable).
        L (nn.Parameter): Lower triangular matrix with ones on the diagonal.
        S (nn.Parameter): Diagonal scaling parameters.
        U (nn.Parameter): Upper triangular matrix with zeros on the diagonal.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # Initialize a random orthogonal matrix.
        Q = torch.nn.init.orthogonal_(torch.randn(dim, dim))
        # Perform LU decomposition: Q = P @ L @ U.
        P, L, U = torch.linalg.lu(Q)
        # Register the permutation matrix as a buffer (non-learnable).
        self.register_buffer("P", P)
        # L, S (diagonal of U), and U (strict upper triangular part) are learnable.
        self.L = nn.Parameter(L)
        self.S = nn.Parameter(U.diag())
        self.U = nn.Parameter(torch.triu(U, diagonal=1))

    def _assemble_W(self) -> torch.Tensor:
        """
        Assembles the full weight matrix W from the LU parameters.
        
        Returns:
            torch.Tensor: The weight matrix W.
        """
        # Construct L with ones on the diagonal.
        L = torch.tril(self.L, diagonal=-1) + torch.diag(torch.ones(self.dim, device=self.L.device, dtype=self.L.dtype))
        # Upper part of U (excluding diagonal) is already stored.
        U = torch.triu(self.U, diagonal=1)
        # Combine P, L, and U with the scaling diagonal to form W.
        W = self.P @ L @ (U + torch.diag(self.S))
        return W

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the 1x1 convolution: z = x @ W, and computes the log-determinant.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            tuple: Transformed tensor and log-determinant.
        """
        # Assemble the weight matrix.
        W = self._assemble_W()
        # Convolve input using matrix multiplication.
        z = x @ W
        # The log-determinant is the sum of log(abs(S)).
        log_det = torch.sum(torch.log(torch.abs(self.S)))
        return z, log_det

    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the inverse of the 1x1 convolution.
        
        Args:
            z (torch.Tensor): Transformed tensor.
        
        Returns:
            tuple: Reconstructed input tensor and inverse log-determinant.
        """
        # Assemble weight matrix and compute its inverse.
        W = self._assemble_W()
        W_inv = torch.linalg.inv(W)
        # Apply inverse convolution.
        x = z @ W_inv
        # Inverse log-determinant is negative of forward log-determinant.
        log_det = -torch.sum(torch.log(torch.abs(self.S)))
        return x, log_det

class NormalizingFlow(nn.Module):
    """
    A sequence of normalizing flows. This class encapsulates multiple flow transformations
    and accumulates their log-determinants.
    
    Attributes:
        flows (ModuleList): List of individual flow modules.
    """
    def __init__(self, flows):
        super().__init__()
        # Store flows in a ModuleList for proper registration.
        self.flows = nn.ModuleList(flows)

    def forward(self, x: torch.Tensor) -> tuple[list, torch.Tensor]:
        """
        Applies a sequence of flow transformations.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            tuple: A list of intermediate outputs (including the input) and the total log-determinant.
        """
        m, _ = x.shape
        # Initialize total log-determinant.
        log_det = torch.zeros(m, device=x.device, dtype=x.dtype)
        # Store the input as the first element.
        zs = [x]
        # Apply each flow sequentially.
        for flow in self.flows:
            x, ld = flow.forward(x)
            log_det += ld
            zs.append(x)
        return zs, log_det

    def inverse(self, z: torch.Tensor) -> tuple[list, torch.Tensor]:
        """
        Reconstructs the input by applying the inverse of each flow in reverse order.
        
        Args:
            z (torch.Tensor): Final latent representation.
        
        Returns:
            tuple: A list of intermediate reconstructions and the total inverse log-determinant.
        """
        m, _ = z.shape
        log_det = torch.zeros(m, device=z.device, dtype=z.dtype)
        xs = [z]
        # Reverse the flow order for inversion.
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z)
            log_det += ld
            xs.append(z)
        return xs, log_det

class NormalizingFlowModel(nn.Module):
    """
    A Normalizing Flow Model combines a prior distribution with a flow.
    It computes the log-probability of data under the transformed space.
    
    Attributes:
        prior: A probability distribution with a log_prob and sample method.
        flow (NormalizingFlow): The sequence of flow transformations.
    """
    def __init__(self, prior, flows):
        super().__init__()
        self.prior = prior
        self.flow = NormalizingFlow(flows)
    
    def forward(self, x: torch.Tensor) -> tuple[list, torch.Tensor, torch.Tensor]:
        """
        Computes the forward pass of the model.
        Applies the flow to x, computes the prior log-probability, and sums with the flow's log-determinant.
        
        Args:
            x (torch.Tensor): Input data.
        
        Returns:
            tuple: List of intermediate flow outputs, the prior log-probability, and the flow's log-determinant.
        """
        zs, log_det = self.flow.forward(x)
        # Compute log-probability under the prior for the final transformed variable.
        prior_logprob = self.prior.log_prob(zs[-1]).view(x.size(0), -1).sum(1)
        return zs, prior_logprob, log_det

    def inverse(self, z: torch.Tensor) -> tuple[list, torch.Tensor]:
        """
        Inverts the flow transformation starting from a latent sample.
        
        Args:
            z (torch.Tensor): Latent variable sampled from the prior.
        
        Returns:
            tuple: List of reconstructed variables and cumulative inverse log-determinant.
        """
        xs, log_det = self.flow.inverse(z)
        return xs, log_det
    
    def sample(self, num_samples: int) -> list:
        """
        Generates samples by drawing from the prior and applying the inverse flow.
        
        Args:
            num_samples (int): Number of samples to generate.
        
        Returns:
            list: Generated samples in the data space.
        """
        # Sample from the prior distribution.
        z = self.prior.sample((num_samples,))
        # Invert the flow to obtain data samples.
        xs, _ = self.flow.inverse(z)
        return xs
