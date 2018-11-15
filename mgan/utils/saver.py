import os
import shutil
import torch
from warnings import warn

class Saver:
    def __init__(self, path):
        self.path = path
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            warn("{path} doesn't exist. Creating.".format(path=path))

    def checkpoint_trainer(self, trainer, is_best=False):
        for tag, savable in trainer.savable:
            self.checkpoint(savable, tag, is_best)

    def checkpoint(self, payload, tag, is_best=False):
        checkpoint_path = self.get_path(tag)
        with open(checkpoint_path, "wb+") as fp:
            _payload = payload.state_dict()
            torch.save(_payload, fp)

        if is_best:
            best_path = '{prefix}.best'.format(prefix=checkpoint_path)
            shutil.copyfile(checkpoint_path, best_path)

    def get_path(self, tag, is_best=False):
        fname = '{tag}.pt'.format(tag=tag)
        checkpoint_path = os.path.join(self.path, fname)
        return checkpoint_path

    def load(self, dest, tag, is_best=False):
        checkpoint_path = self.get_path(tag)
        if is_best: checkpoint_path = '{prefix}.best'.format(prefix=checkpoint_path)

        if os.path.exists(checkpoint_path):
            payload = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            _payload = dest.state_dict()
            _payload.update(payload)
            dest.load_state_dict(_payload)
        else:
            warn("Error: No Weights loaded.".format(path=checkpoint_path))

    def load_trainer(self, trainer, is_best=False):
        for tag, savable in trainer.savable:
            self.load(savable, tag, is_best)
