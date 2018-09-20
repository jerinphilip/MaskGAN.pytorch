
from . import mask, tokenize
from torch import nn


class Preprocess(nn.Module):

    MASK_REGISTRY = {
        "end": mask.EndMask,
    }

    TOKENIZE_REGISTRY = {
        "space": tokenize.SpaceTokenizer
    }

    def __init__(self, mask, tokenize):
        super().__init__()

        mtype, args, kwargs = self.triplet(mask)
        self.mask = self.MASK_REGISTRY[mtype](*args, **kwargs)

        ttype, args, kwargs = self.triplet(tokenize)
        self.tokenize = self.TOKENIZE_REGISTRY[ttype](*args, **kwargs)

    def forward(self, seq, mask=True):
        if mask: return self.mask(self.tokenize(seq))
        return self.tokenize(seq)

    def triplet(self, payload):
        mtype = payload["type"]
        args = payload.get("args", None) or []
        kwargs = payload.get("kwargs", None) or {}
        return (mtype, args, kwargs)

