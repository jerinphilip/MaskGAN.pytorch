from torch import nn


class Mask(nn.Module):
    mask_token = '__<m>__'

class EndMask(Mask):
    def __init__(self, n_chars):
        super().__init__()
        self.n_chars = n_chars
    
    def forward(self, xs):
        # x is supposed to be a set of tokens
        for i in range(self.n_chars):
            j = i+1
            xs[-j] = self.mask_token
        return xs



