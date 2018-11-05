from torch import nn
import torch
import random

class Mask:
    mask_token = '__<m>__'
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class EndMask(Mask):
    def __init__(self, n_chars):
        super().__init__()
        self.n_chars = n_chars
    
    def forward(self, xs):
        # x is supposed to be a set of tokens
        mask = torch.zeros(self.n_chars)
        for i in range(self.n_chars):
            j = i+1
            xs[-j] = self.mask_token
            mask[-j] = 1
        return (xs, mask)



class StochasticMask(Mask):
    def __init__(self, probability):
        self.p = probability
        self.r = random.Random(42)

    def forward(self, xs):
        ys = []
        mask_count = 0
        mask = []
        for i, x in enumerate(xs):
            if self.r.random() < self.p:
                mask_count += 1
                ys.append(self.mask_token)
                mask.append(True)
            else:
                ys.append(x)
                mask.append(False)

        mask = torch.Tensor(mask)
        return (ys, mask)


