from torch import nn

class DistributedModel(nn.Module):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(*args, **kwargs):
        raise NotImplementedError

