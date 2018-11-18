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
        n_chars = self.n_chars
        mask = torch.zeros(len(xs))
        for i in range(n_chars):
            j = i+1
            xs[-j] = self.mask_token
            mask[-j] = 1
        return (xs, mask)

class ContiguousRandom(Mask):
    def __init__(self, n_chars):
        super().__init__()
        self.n_chars = n_chars
        self.r = random.Random(42)

    def forward(self, xs):
        n_chars = self.n_chars
        mask = torch.zeros(len(xs))
        start = self.r.randint(3, len(xs)-n_chars-1)
        for i in range(start, start+n_chars):
            xs[i] = self.mask_token
            mask[i] = 1
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


