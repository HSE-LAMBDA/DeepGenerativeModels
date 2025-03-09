"""
Neural Spline Flows: Coupling and Autoregressive variants.

This implementation follows the method proposed by Durkan et al. (https://arxiv.org/abs/1906.04032)
and is adapted from the code at:
https://github.com/tonyduan/normalizing-flows/blob/master/nf/flows.py

The code implements Rational Quadratic Spline (RQS) transforms which are used in normalizing flows.
It supports both forward and inverse transformations and computes the log absolute determinant
of the Jacobian for change-of-variable computations.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from nflib.nets import MLP  # Import a multilayer perceptron (MLP) network for parameter generation

# Default constants to ensure numerical stability in the spline transformation
DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

def searchsorted(bin_locations: torch.Tensor, inputs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Finds the bin indices for each input value given the cumulative bin boundaries.

    Args:
        bin_locations (torch.Tensor): Tensor of cumulative bin locations. Expected to have the last element perturbed.
        inputs (torch.Tensor): Tensor of input values for which to determine the bin index.
        eps (float): A small epsilon value to avoid precision issues at the boundary.

    Returns:
        torch.Tensor: The indices of the bins where each input belongs.
    """
    # Slightly adjust the last bin boundary to include edge cases
    bin_locations[..., -1] += eps
    # Count the number of bin boundaries each input surpasses and subtract 1 to get the bin index
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1

def unconstrained_RQS(inputs: torch.Tensor,
                      unnormalized_widths: torch.Tensor,
                      unnormalized_heights: torch.Tensor,
                      unnormalized_derivatives: torch.Tensor,
                      inverse: bool = False,
                      tail_bound: float = 1.,
                      min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
                      min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
                      min_derivative: float = DEFAULT_MIN_DERIVATIVE) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the unconstrained rational quadratic spline (RQS) transform to the inputs.
    For inputs outside the specified tail bound, the transform is identity.

    Args:
        inputs (torch.Tensor): Input tensor to be transformed.
        unnormalized_widths (torch.Tensor): Unnormalized widths for the spline bins.
        unnormalized_heights (torch.Tensor): Unnormalized heights for the spline bins.
        unnormalized_derivatives (torch.Tensor): Unnormalized derivatives at the bin edges.
        inverse (bool): Flag indicating whether to perform the inverse transformation.
        tail_bound (float): Bound for inputs; values outside [-tail_bound, tail_bound] are left unchanged.
        min_bin_width (float): Minimum width for each bin to ensure stability.
        min_bin_height (float): Minimum height for each bin to ensure stability.
        min_derivative (float): Minimum derivative value for numerical stability.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - outputs: Transformed output tensor.
            - logabsdet: Log absolute determinant of the Jacobian for the transform.
    """
    # Determine which inputs are within the transformation interval ([-tail_bound, tail_bound])
    inside_intvl_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_intvl_mask

    # Initialize outputs and log-determinant tensor with zeros
    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    # Pad unnormalized derivatives to include boundary conditions
    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = np.log(np.exp(1 - min_derivative) - 1)
    # Set boundary derivatives to a constant to enforce the minimum derivative constraint
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    # For inputs outside the interval, apply the identity transform
    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0

    # For inputs inside the interval, apply the RQS transform
    outputs[inside_intvl_mask], logabsdet[inside_intvl_mask] = RQS(
        inputs=inputs[inside_intvl_mask],
        unnormalized_widths=unnormalized_widths[inside_intvl_mask, :],
        unnormalized_heights=unnormalized_heights[inside_intvl_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_intvl_mask, :],
        inverse=inverse,
        left=-tail_bound, right=tail_bound,
        bottom=-tail_bound, top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )
    return outputs, logabsdet

def RQS(inputs: torch.Tensor,
        unnormalized_widths: torch.Tensor,
        unnormalized_heights: torch.Tensor,
        unnormalized_derivatives: torch.Tensor,
        inverse: bool = False,
        left: float = 0.,
        right: float = 1.,
        bottom: float = 0.,
        top: float = 1.,
        min_bin_width: float = DEFAULT_MIN_BIN_WIDTH,
        min_bin_height: float = DEFAULT_MIN_BIN_HEIGHT,
        min_derivative: float = DEFAULT_MIN_DERIVATIVE) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies the rational quadratic spline (RQS) transformation to the input tensor.

    The transformation maps inputs from the interval [left, right] to [bottom, top] (or vice versa
    for the inverse) using a piecewise rational quadratic function determined by the provided
    unnormalized parameters.

    Args:
        inputs (torch.Tensor): Input tensor to be transformed. Must lie within [left, right].
        unnormalized_widths (torch.Tensor): Unnormalized widths for the spline bins.
        unnormalized_heights (torch.Tensor): Unnormalized heights for the spline bins.
        unnormalized_derivatives (torch.Tensor): Unnormalized derivatives at the spline bin boundaries.
        inverse (bool): If True, computes the inverse transformation.
        left (float): Left boundary of the input domain.
        right (float): Right boundary of the input domain.
        bottom (float): Lower bound of the output domain.
        top (float): Upper bound of the output domain.
        min_bin_width (float): Minimum allowed width for each bin.
        min_bin_height (float): Minimum allowed height for each bin.
        min_derivative (float): Minimum allowed derivative at the bin boundaries.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - outputs: Transformed tensor.
            - logabsdet: Log absolute determinant of the Jacobian of the transformation.
    """
    # Validate input domain
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError("Input outside domain")

    num_bins = unnormalized_widths.shape[-1]

    # Ensure that the minimal bin widths and heights are feasible given the number of bins
    if min_bin_width * num_bins > 1.0:
        raise ValueError('Minimal bin width too large for the number of bins')
    if min_bin_height * num_bins > 1.0:
        raise ValueError('Minimal bin height too large for the number of bins')

    # Normalize widths using softmax and adjust them to satisfy the minimum width constraint
    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    # Pad cumulative widths with a zero at the beginning for boundary alignment
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    # Normalize derivatives using softplus to ensure positivity and add the minimum derivative
    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    # Normalize heights using softmax and adjust them to satisfy the minimum height constraint
    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    # Pad cumulative heights with a zero at the beginning for boundary alignment
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    # Determine the bin index for each input value based on whether we are inverting the transform
    if inverse:
        # For the inverse, find the bin using cumulative heights
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        # For the forward transform, find the bin using cumulative widths
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    # Gather the spline parameters for the appropriate bin for each input
    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]
    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths  # Local slope between heights and widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]
    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]
    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        # Inverse transformation: solve a quadratic equation for the fractional bin position 'root'
        a = (((inputs - input_cumheights) * (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
             + input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives - (inputs - input_cumheights)
             * (input_derivatives + input_derivatives_plus_one - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)

        # Compute the discriminant of the quadratic equation
        discriminant = b.pow(2) - 4 * a * c
        if not (discriminant >= 0).all():
            raise ValueError("Negative discriminant encountered in RQS inverse")
        # Solve the quadratic equation for the fractional position within the bin
        root = (2 * c) / (-b - torch.sqrt(discriminant))
        # Map the fractional position back to the original input scale
        outputs = root * input_bin_widths + input_cumwidths

        # Compute derivative components required for the log absolute determinant of the Jacobian
        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2) + 2 * input_delta * theta_one_minus_theta +
            input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        # Return the inverse log-det (negative sign) to account for inversion
        return outputs, -logabsdet
    else:
        # Forward transformation: compute the fractional position 'theta' within the bin
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        # Numerator and denominator for the spline function calculation
        numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta)
        # Map the fractional position to the output space
        outputs = input_cumheights + numerator / denominator

        # Compute the derivative of the transformation
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2) + 2 * input_delta * theta_one_minus_theta +
            input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, logabsdet

class NSF_AR(nn.Module):
    """
    Neural Spline Flow (Autoregressive variant).

    This module applies a series of 1-dimensional spline transformations sequentially over the input dimensions.
    For each dimension, the transformation parameters are generated either from a global parameter (for the first
    dimension) or from an autoregressive network that conditions on the previously transformed dimensions.

    Attributes:
        dim (int): Dimensionality of the input.
        K (int): Number of bins for the spline transformation.
        B (int): Tail bound, determines the input domain [-B, B] where the spline is active.
        layers (nn.ModuleList): List of autoregressive networks for dimensions > 1.
        init_param (nn.Parameter): Global parameters used for the first input dimension.
    """
    def __init__(self, dim: int, K: int = 5, B: int = 3, hidden_dim: int = 8, base_network=MLP):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.layers = nn.ModuleList()
        # Global initial parameters for the first dimension transformation
        self.init_param = nn.Parameter(torch.Tensor(3 * K - 1))
        # Create autoregressive networks for each subsequent dimension that condition on previous dimensions
        for i in range(1, dim):
            self.layers.append(base_network(i, 3 * K - 1, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes the global parameter uniformly."""
        init.uniform_(self.init_param, -0.5, 0.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the forward autoregressive spline transformation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - z: Transformed output tensor.
                - log_det: Log-determinant of the Jacobian of the transformation.
        """
        z = torch.zeros_like(x)
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        # Process each dimension sequentially
        for i in range(self.dim):
            if i == 0:
                # For the first dimension, use the global initial parameters
                init_param = self.init_param.expand(x.shape[0], 3 * self.K - 1)
                W, H, D = torch.split(init_param, self.K, dim=1)
            else:
                # For subsequent dimensions, generate parameters from the autoregressive network conditioned on previous dimensions
                out = self.layers[i - 1](x[:, :i])
                W, H, D = torch.split(out, self.K, dim=1)
            # Normalize widths and heights to enforce proper constraints
            W, H = torch.softmax(W, dim=1), torch.softmax(H, dim=1)
            W, H = 2 * self.B * W, 2 * self.B * H
            # Ensure derivatives are positive by applying softplus
            D = F.softplus(D)
            # Apply the rational quadratic spline transform to the current dimension
            z[:, i], ld = unconstrained_RQS(x[:, i], W, H, D, inverse=False, tail_bound=self.B)
            log_det += ld
        return z, log_det

    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the inverse of the autoregressive spline transformation.

        Args:
            z (torch.Tensor): Transformed tensor of shape (batch_size, dim).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - x: Inverse transformed tensor (original input space).
                - log_det: Log-determinant of the inverse transformation.
        """
        x = torch.zeros_like(z)
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        # Process each dimension sequentially in the same autoregressive order
        for i in range(self.dim):
            if i == 0:
                init_param = self.init_param.expand(z.shape[0], 3 * self.K - 1)
                W, H, D = torch.split(init_param, self.K, dim=1)
            else:
                out = self.layers[i - 1](x[:, :i])
                W, H, D = torch.split(out, self.K, dim=1)
            W, H = torch.softmax(W, dim=1), torch.softmax(H, dim=1)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            # Invert the spline transformation for the current dimension
            x[:, i], ld = unconstrained_RQS(z[:, i], W, H, D, inverse=True, tail_bound=self.B)
            log_det += ld
        return x, log_det

class NSF_CL(nn.Module):
    """
    Neural Spline Flow (Coupling variant).

    This module splits the input into two halves and applies spline transformations using coupling layers.
    Two separate networks (f1 and f2) generate the transformation parameters for each half, enabling efficient
    computation of the inverse transformation while preserving a tractable Jacobian.
    
    Attributes:
        dim (int): Dimensionality of the input. Should be even.
        K (int): Number of bins for the spline transformation.
        B (int): Tail bound, determines the input domain [-B, B] where the spline is active.
        f1 (nn.Module): Network for generating parameters to transform the upper half conditioned on the lower half.
        f2 (nn.Module): Network for generating parameters to transform the lower half conditioned on the upper half.
    """
    def __init__(self, dim: int, K: int = 5, B: int = 3, hidden_dim: int = 8, base_network=MLP):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        # f1 processes the lower half to generate parameters for the upper half transformation
        self.f1 = base_network(dim // 2, (3 * K - 1) * (dim // 2), hidden_dim)
        # f2 processes the upper half to generate parameters for the lower half transformation
        self.f2 = base_network(dim // 2, (3 * K - 1) * (dim // 2), hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the forward coupling-based spline transformation.

        The input is split into two halves. The lower half conditions the transformation of the upper half,
        and then the upper half conditions the transformation of the lower half.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Transformed tensor with the same shape as x.
                - Log-determinant of the Jacobian of the overall transformation.
        """
        log_det = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        # Split the input into lower and upper halves
        lower, upper = x[:, :self.dim // 2], x[:, self.dim // 2:]
        
        # Transform the upper half conditioned on the lower half using network f1
        out = self.f1(lower).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim=2)
        W, H = torch.softmax(W, dim=2), torch.softmax(H, dim=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_RQS(upper, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim=1)
        
        # Transform the lower half conditioned on the transformed upper half using network f2
        out = self.f2(upper).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim=2)
        W, H = torch.softmax(W, dim=2), torch.softmax(H, dim=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        lower, ld = unconstrained_RQS(lower, W, H, D, inverse=False, tail_bound=self.B)
        log_det += torch.sum(ld, dim=1)
        
        # Concatenate the transformed halves to form the output
        return torch.cat([lower, upper], dim=1), log_det

    def inverse(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the inverse coupling-based spline transformation.

        In the inverse transformation, the roles of f1 and f2 are reversed:
        first inverting the transformation on the lower half conditioned on the upper half,
        and then inverting the transformation on the upper half conditioned on the recovered lower half.

        Args:
            z (torch.Tensor): Transformed tensor of shape (batch_size, dim).

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - Inverse transformed tensor (original input space).
                - Log-determinant of the inverse transformation.
        """
        log_det = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        # Split the transformed input into lower and upper halves
        lower, upper = z[:, :self.dim // 2], z[:, self.dim // 2:]
        
        # Invert the transformation on the lower half conditioned on the upper half using network f2
        out = self.f2(upper).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim=2)
        W, H = torch.softmax(W, dim=2), torch.softmax(H, dim=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        lower, ld = unconstrained_RQS(lower, W, H, D, inverse=True, tail_bound=self.B)
        log_det += torch.sum(ld, dim=1)
        
        # Invert the transformation on the upper half conditioned on the recovered lower half using network f1
        out = self.f1(lower).reshape(-1, self.dim // 2, 3 * self.K - 1)
        W, H, D = torch.split(out, self.K, dim=2)
        W, H = torch.softmax(W, dim=2), torch.softmax(H, dim=2)
        W, H = 2 * self.B * W, 2 * self.B * H
        D = F.softplus(D)
        upper, ld = unconstrained_RQS(upper, W, H, D, inverse=True, tail_bound=self.B)
        log_det += torch.sum(ld, dim=1)
        
        # Concatenate the recovered halves to form the original input
        return torch.cat([lower, upper], dim=1), log_det
