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

    def checkpoint(self, model, opt, tag, is_best=False):
        _payload = {}
        _payload["model"] = model.state_dict()
        if opt is not None:
            _payload["opt"] = opt.state_dict()

        checkpoint_path = self.get_path(tag)
        with open(checkpoint_path, "wb+") as fp:
            torch.save(_payload, fp)

        if is_best:
            best_path = '{prefix}.best'.format(prefix=checkpoint_path)
            shutil.copyfile(checkpoint_path, best_path)

    def get_path(self, tag, is_best=False):
        fname = '{tag}.ckpt'.format(tag=tag)
        checkpoint_path = os.path.join(self.path, fname)
        return checkpoint_path


    def load(self, model, opt, tag, is_best=False):
        checkpoint_path = self.get_path(tag)

        if is_best:
            checkpoint_path = '{prefix}.best'.format(prefix=checkpoint_path)

        if os.path.exists(checkpoint_path):
            _payload = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            model.load_state_dict(_payload["model"])
            if opt is not None:
                opt.load_state_dict(_payload["opt"])
        else:
            warn("FileDoesNotExist: {path}, no weights loaded.".format(path=checkpoint_path))
            
