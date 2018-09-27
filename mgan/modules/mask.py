from torch import nn
from random import uniform, random


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
        for i in range(self.n_chars):
            j = i+1
            xs[-j] = self.mask_token
        return xs



class StochasticMask(Mask):
    def __init__(self, probability):
        self.p = probability

    def forward(self, xs):
        ys = []
        for i, x in enumerate(xs):
            if random() < self.p:
                ys.append(self.mask_token)
            else:
                ys.append(x)
        return ys


