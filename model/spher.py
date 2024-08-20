import torch.nn as nn
import torch
class Spher(nn.Module):
    """
    """
    def __init__(self, batch_size=16, dim=128, C=None):
        """
        """
        super(Spher, self).__init__()
        self.C = nn.parameter.Parameter(C)
        self.R = nn.parameter.Parameter(torch.FloatTensor(0.0))
        self.nu = 0.1
    def forward(self, x):
        dist = torch.sum((x - self.C) ** 2, dim=1)
        scores = dist - self.R ** 2
        loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        return loss