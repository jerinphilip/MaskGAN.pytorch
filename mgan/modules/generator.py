
import torch.nn.functional as F
from torch import nn

class Generator(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args):
        net_output = self.model(*args)
        logits = net_output[0].float()
        return F.log_softmax(logits, dim=-1)
