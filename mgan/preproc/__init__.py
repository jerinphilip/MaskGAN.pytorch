from . import mask, tokenize
import torch

class Preprocess:
    def __init__(self, mask, tokenize, truncate=-1):
        self.truncate = truncate
        self.mask = mask
        self.tokenize = tokenize

    def __call__(self, seq, mask=True):
        tokenized = self.tokenize(seq)
        if self.truncate > 0:
            final_length = min(len(tokenized), self.truncate)
            tokenized = tokenized[:final_length]
            while len(tokenized) < final_length:
                tokenized.append('<pad>')

        if mask: tokens, mask = self.mask(tokenized)
        else: tokens, mask = tokenized, torch.zeros(len(tokenized))
        return tokens, mask
