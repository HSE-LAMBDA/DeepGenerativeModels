"""
Various helper network modules.

This module includes utility classes to facilitate building neural network components.
It provides:
- LeafParam: A parameter tensor independent of the input.
- PositionalEncoder: A module to add sinusoidal positional encodings.
- MLP: A simple multi-layer perceptron.
- PosEncMLP: An MLP that applies positional encoding to its inputs.
- ARMLP: An auto-regressive MLP implemented via the MADE architecture.
"""

import torch
from torch import nn

from nflib.made import MADE  # MADE is a masked autoencoder for distribution estimation.

class LeafParam(nn.Module):
    """
    Module that outputs a learnable parameter tensor independent of the input.
    
    This module initializes a parameter 'p' and, during the forward pass, expands it
    to match the batch size of the given input. It is useful when a constant parameter
    needs to be learned and applied across different inputs.
    """
    def __init__(self, n: int):
        """
        Initialize the LeafParam module.
        
        Args:
            n (int): The number of parameters to learn.
        """
        super().__init__()
        # Create a learnable parameter with shape (1, n) initialized to zeros.
        self.p = nn.Parameter(torch.zeros(1, n))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that returns the expanded parameter.
        
        Args:
            x (torch.Tensor): Input tensor where only the batch size is considered.
        
        Returns:
            torch.Tensor: The learned parameter expanded to shape (batch_size, n).
        """
        # Expand 'p' along the batch dimension to match the input's batch size.
        return self.p.expand(x.size(0), self.p.size(1))

class PositionalEncoder(nn.Module):
    """
    Module to apply sinusoidal positional encoding to inputs.
    
    This encoder takes an input tensor and for each feature computes sine and cosine
    transformations at multiple specified frequencies. The outputs are concatenated,
    expanding each input dimension to enhance the representation with periodic features.
    """
    def __init__(self, freqs=(0.5, 1, 2, 4, 8)):
        """
        Initialize the PositionalEncoder module.
        
        Args:
            freqs (tuple, optional): Frequencies for the sine and cosine functions.
                                     Defaults to (0.5, 1, 2, 4, 8).
        """
        super().__init__()
        self.freqs = freqs
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that applies positional encoding.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, features).
            
        Returns:
            torch.Tensor: Positional encoded tensor with shape 
                          (batch_size, features * len(freqs) * 2).
        """
        # Compute sine transformations for each frequency.
        sines = [torch.sin(x * f) for f in self.freqs]
        # Compute cosine transformations for each frequency.
        coses = [torch.cos(x * f) for f in self.freqs]
        # Concatenate sine and cosine results along the feature dimension.
        out = torch.cat(sines + coses, dim=1)
        return out

class MLP(nn.Module):
    """
    A simple 4-layer Multi-Layer Perceptron (MLP).
    
    The architecture consists of four linear layers interleaved with LeakyReLU activation functions.
    The network layout is:
        Linear -> LeakyReLU -> Linear -> LeakyReLU -> Linear -> LeakyReLU -> Linear.
    """
    def __init__(self, nin: int, nout: int, nh: int):
        """
        Initialize the MLP module.
        
        Args:
            nin (int): Number of input features.
            nout (int): Number of output features.
            nh (int): Number of units in each hidden layer.
        """
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
        """
        Forward pass of the MLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, nin).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, nout).
        """
        return self.net(x)

class PosEncMLP(nn.Module):
    """
    A Position Encoded MLP that integrates positional encoding into an MLP.
    
    This module first transforms each input feature using a fixed sine and cosine encoding
    (via the PositionalEncoder), effectively expanding each feature dimension. The expanded
    representation is then processed by a standard MLP.
    """
    def __init__(self, nin: int, nout: int, nh: int, freqs=(0.5, 1, 2, 4, 8)):
        """
        Initialize the PosEncMLP module.
        
        Args:
            nin (int): Number of original input features.
            nout (int): Number of output features.
            nh (int): Number of hidden units for the MLP.
            freqs (tuple, optional): Frequencies used in positional encoding.
                                     Defaults to (0.5, 1, 2, 4, 8).
        """
        super().__init__()
        self.net = nn.Sequential(
            # Apply positional encoding to expand each input dimension.
            PositionalEncoder(freqs),
            # Process the expanded input with an MLP.
            # The input size for the MLP is nin * (len(freqs) * 2) since each original feature
            # is transformed into 2 values per frequency.
            MLP(nin * len(freqs) * 2, nout, nh),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PosEncMLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, nin).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, nout).
        """
        return self.net(x)

class ARMLP(nn.Module):
    """
    Auto-regressive MLP implemented using the MADE architecture.
    
    The MADE (Masked Autoencoder for Distribution Estimation) ensures an auto-regressive 
    property by applying masks to the network layers, making each output depend only on 
    previous inputs. This is useful in probabilistic modeling and sequential data processing.
    """
    def __init__(self, nin: int, nout: int, nh: int):
        """
        Initialize the ARMLP module.
        
        Args:
            nin (int): Number of input features.
            nout (int): Number of output features.
            nh (int): Number of hidden units in each hidden layer.
        """
        super().__init__()
        self.net = MADE(nin, [nh, nh, nh], nout, num_masks=1, natural_ordering=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ARMLP.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, nin).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, nout) computed using an auto-regressive approach.
        """
        return self.net(x)
