from torch import nn
import torch
import random

class Mask:
    mask_token = '__<m>__'
    def __call__(self, n):
        idxs = self.forward(n)

        # Verify indices are okay.
        assert ( len(idxs) < n)
        valid_set = set(list(range(n)))
        for i in idxs:
            assert(i in valid_set)

        return idxs


class EndMask(Mask):
    def __init__(self, n_chars):
        super().__init__()
        self.n_chars = n_chars
    
    def forward(self, n):
        # x is supposed to be a set of tokens
        n_chars = self.n_chars
        idxs = []
        for i in range(n_chars):
            idxs.append(n-i-1)
        return idxs


class ContiguousRandom(Mask):
    def __init__(self, n_chars):
        super().__init__()
        self.n_chars = n_chars
        self.r = random.Random(42)

    def forward(self, n):
        n_chars = self.n_chars
        start = self.r.randint(1, n-n_chars-1)
        assert ( start + n_chars <= n)
        idxs = []
        for i in range(start, start+n_chars):
            idxs.append(i)
        return idxs



class StochasticMask(Mask):
    def __init__(self, probability):
        self.p = probability
        self.r = random.Random(42)

    def forward(self, n):
        # TODO(jerin), convert into proper bernoulli?
        idxs = []
        for i in range(n):
            if self.r.random() < self.p:
                idxs.append(i)
        return idxs


