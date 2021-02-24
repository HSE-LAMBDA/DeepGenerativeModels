import torch
from torch import nn
from .utils import compute_gradient_penalty, permute_labels

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        # YOUR CODE
        
    def forward(self, x, labels):
        # YOUR CODE

        
class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        # YOUR CODE
        
    def forward(self, x):
        # YOUR CODE

        
class StarGAN:
    def __init__(self):
        self.G = Generator()
        self.D = Critic()
        
        # YOUR CODE
        
    def train(self):
        self.G.train()
        self.D.train()
        
    def eval(self):
        self.G.eval()
        self.D.eval()

    def to(self, device):
        self.D.to(device)
        self.G.to(device)
        
    def trainG(self, image, label):
        # YOUR CODE
        
    def trainD(self, image, label):
        # YOUR CODE

    def generate(self, image, label):
        # YOUR CODE