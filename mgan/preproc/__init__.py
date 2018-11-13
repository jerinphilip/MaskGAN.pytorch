
from . import mask, tokenize
from torch import nn
import torch


class Preprocess(nn.Module):

    MASK_REGISTRY = {
        "end": mask.EndMask,
        "random": mask.StochasticMask
    }

    TOKENIZE_REGISTRY = {
        "space": tokenize.SpaceTokenizer,
        "spm": tokenize.SentencePieceTokenizer
    }

    def __init__(self, mask, tokenize):
        super().__init__()
        mtype, args, kwargs = self.triplet(mask)
        self.mask = self.MASK_REGISTRY[mtype](*args, **kwargs)

        ttype, args, kwargs = self.triplet(tokenize)
        self.tokenize = self.TOKENIZE_REGISTRY[ttype](*args, **kwargs)

    def forward(self, seq, mask=True):
        tokenized = self.tokenize(seq)
        if mask: 
            tokens, mask = self.mask(tokenized)
        else:
            tokens, mask = tokenized, torch.zeros(len(tokenized))
        return tokens, mask

    def triplet(self, payload):
        mtype = payload["type"]
        args = payload.get("args", None) or []
        kwargs = payload.get("kwargs", None) or {}
        return (mtype, args, kwargs)

