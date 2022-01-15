import torch
import numpy as np

class ForwardOperator(torch.nn.Module):
    """
    Forward operator
    """
    def __init__(self, equation=4):
        super(ForwardOperator, self).__init__()
        self.equation = equation
        if equation == 4 or equation == 6:
            self.noise_dist = torch.distributions.gamma.Gamma(1.0, 1.0/0.3)
        elif equation == 5:
            self.noise_dist = torch.distributions.normal.Normal(0., 0.05)
    def forward(self, x):
        if self.equation == 4:
            return torch.tanh(x) + self.noise_dist.sample(x.shape)
        elif self.equation == 5:
            return torch.tanh(x + self.noise_dist.sample(x.shape))
        elif self.equation == 6:
            return torch.tanh(x) * self.noise_dist.sample(x.shape)

class Mask(torch.nn.Module):
    """
    Mask operator
    """
    def __init__(self, shape, device):
        super(Mask, self).__init__()

        self.mask = torch.ones(shape, device=device)
        self.mask[3:-3, 3:-3] = 0.0

    def forward(self, x):
        return self.mask * x