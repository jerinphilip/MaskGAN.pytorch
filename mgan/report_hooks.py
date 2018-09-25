from pprint import pprint
import sys
import torch

class DevNull:
    def __call__(self, *args, **kwargs):
        pass

class Log:
    def __init__(self, log_file=sys.stderr):
        self.log_file = log_file

    def __call__(self, _dict):
        pprint(_dict, self.log_file)


class TransBatchCompileHook:
    def __init__(self):
        self.state = {}

    def __call__(self, _dict, step=False):
        if not step:
            self.state.update(_dict)
        else:
            self._handle(_dict)


    def _handle(self, _dict):
        self.update("attns", _dict)
        self.update("preds", _dict)

    def update(self, key, _dict):
        if key  not in self.state:
            self.state[key] = _dict[key].unsqueeze(0)
        else:
            a = self.state[key]
            a_t = _dict[key].unsqueeze(0)
            self.state[key] = torch.cat([a, a_t], dim=0)

    def __len__(self):
        T, B, H = self.state["inputs"].size()
        return B

    def __getitem__(self, idx):
        assert(idx < len(self))
        inputs = self.state["inputs"][:, idx, :]
        if self.state["truths"] is not None:
            truths = self.state["truths"][:, idx, :]
        else:
            truths = None
        preds = self.state["preds"][:, idx]
        attns = self.state["attns"][:, idx, :, :].squeeze(1)
        return (inputs, truths, preds, attns)




