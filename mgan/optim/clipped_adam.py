import torch
from torch.nn.utils.clip_grad import clip_grad_norm_


class ClippedAdam(torch.optim.Adam):
    def __init__(self, parameters, *args, **kwargs):
        super().__init__(parameters, *args, **kwargs)
        self.clip_value = None
        self._parameters = parameters

    def set_clip(self, clip_value):
        self.clip_value = clip_value

    def step(self, *args, **kwargs):
        assert (self.clip_value is not None)
        clip_grad_norm_(self._parameters, self.clip_value)
        super().step(*args, **kwargs)
