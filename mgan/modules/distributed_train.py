from .distributed_model import DistributedModel
import torch
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn.parallel import DataParallel

class DistributedTrain:
    def __init__(self, model, opt, device=torch.device("cuda")):
        assert(isinstance(model, DistributedModel))
        self.model = model
        self.model = self.model.to(device)
        self.distributed_model = DataParallel(model)
        self.opt = opt
        self.device = device

    def parallel(self):
        self.model = self.model.to(self.device)

    def logits(self):
        umodel = self.distributed_model.module.underlying_model()
        return DataParallel(umodel)

    def __call__(self, *args, **kwargs):
        self.opt.zero_grad()
        loss, samples = self.distributed_model(*args, **kwargs)
        loss = loss.mean()
        loss.backward()
        self.opt.step()
        return (loss.item(), samples)

    def eval(self, *args, **kwargs):
        with torch.no_grad():
            loss, samples = self.distributed_model(*args, **kwargs)
            loss = loss.mean()
            return (loss.item(), samples)
