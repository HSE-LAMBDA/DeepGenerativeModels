"""
Implements a Masked Autoregressive MLP, where carefully constructed binary masks
over weights ensure the autoregressive property. This design enables the model
to enforce dependencies only on preceding variables in a specified ordering.

Source: https://github.com/karpathy/pytorch-made
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

class MaskedLinear(nn.Linear):
    """
    Linear layer with a configurable binary mask applied to its weights.
    
    The mask ensures that the autoregressive property is maintained by selectively
    zeroing out connections that violate the dependency ordering.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Initializes the MaskedLinear layer.
        
        Parameters:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            bias (bool): If True, includes a bias term.
        """
        super().__init__(in_features, out_features, bias)
        # Register a non-trainable buffer for the mask; initialize with ones.
        self.register_buffer('mask', torch.ones(out_features, in_features))
        
    def set_mask(self, mask: np.ndarray) -> None:
        """
        Sets the binary mask for the layer weights.
        
        Parameters:
            mask (np.ndarray): A binary mask array that defines which connections
                               should be active (1) or inactive (0). The mask is
                               expected to have a shape compatible with the weight matrix.
        """
        # Convert the numpy mask to a float tensor and transpose it to match the weight shape.
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.float32).T))
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MaskedLinear layer.
        
        Parameters:
            input (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: The result of applying a linear transformation with masked weights.
        """
        # Multiply weights by the mask to zero out forbidden connections, then perform the linear operation.
        return F.linear(input, self.mask * self.weight, self.bias)

class MADE(nn.Module):
    """
    Masked Autoencoder for Distribution Estimation (MADE).
    
    This model implements an autoregressive neural network where each output is
    conditionally independent of future inputs, enabling efficient sampling and density estimation.
    """
    def __init__(self, nin: int, hidden_sizes: list, nout: int, num_masks: int = 1, natural_ordering: bool = False):
        """
        Initializes the MADE network.
        
        Parameters:
            nin (int): Number of input features.
            hidden_sizes (list): List of integers representing the number of units in each hidden layer.
            nout (int): Number of output units. Typically, nout is an integer multiple of nin since it may parameterize a 1D distribution.
            num_masks (int): Number of mask orderings for creating an ensemble of models. When set to 1, the mask is not updated after initialization.
            natural_ordering (bool): If True, uses natural ordering (i.e., [0, 1, ..., nin-1]) for dependencies. Otherwise, uses random permutations.
        """
        super().__init__()
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        
        # Ensure that the number of outputs is an integer multiple of the number of inputs.
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"
        
        # Build a simple MLP with MaskedLinear layers and ReLU activations.
        self.net = []
        hs = [nin] + hidden_sizes + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            # Append a MaskedLinear layer followed by a ReLU activation.
            self.net.extend([
                MaskedLinear(h0, h1),
                nn.ReLU(),
            ])
        # Remove the last ReLU since it is not needed after the final linear layer.
        self.net.pop()
        self.net = nn.Sequential(*self.net)
        
        # Store mask configuration parameters.
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0
        
        # Dictionary to store connectivity masks for each layer.
        self.m = {}
        # Generate the initial set of masks.
        self.update_masks()
        
    def update_masks(self) -> None:
        """
        Generates new binary masks for all MaskedLinear layers in the network.
        
        The masks are constructed to enforce an autoregressive ordering of the inputs.
        For each layer, random integers determine the connectivity based on the chosen ordering.
        If num_masks is 1 and masks are already generated, the function will not update them.
        """
        # Skip update if masks already exist and only one ordering is used.
        if self.m and self.num_masks == 1:
            return
        
        L = len(self.hidden_sizes)  # Number of hidden layers.
        rng = np.random.RandomState(self.seed)
        # Update the seed for future mask updates (rotates among available mask orderings).
        self.seed = (self.seed + 1) % self.num_masks
        
        # Assign ordering for the input layer: either natural or random.
        self.m[-1] = np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin)
        
        # For each hidden layer, assign random connectivity numbers that are within valid bounds.
        for l in range(L):
            self.m[l] = rng.randint(self.m[l - 1].min(), self.nin - 1, size=self.hidden_sizes[l])
        
        # Create binary masks for connections between successive layers.
        # The mask ensures that connections only exist from "earlier" to "later" variables.
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        # For the output layer, enforce strict ordering with a less-than condition.
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])
        
        # If there are more outputs than inputs, repeat the final mask accordingly.
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)
        
        # Retrieve all MaskedLinear layers from the network.
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        # Set the computed masks for each corresponding MaskedLinear layer.
        for l, m in zip(layers, masks):
            l.set_mask(m)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MADE network.
        
        Parameters:
            x (torch.Tensor): Input tensor with shape (batch_size, nin)
            
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, nout) after processing through the network.
        """
        return self.net(x)
