import torch
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.nn.parallel import DataParallel

class DistributedTrain:
    def __init__(self, model, device=torch.device("cuda")):
        # self.device_ids = list(range(torch.cuda.device_count()))
        # self.model = (DataParallel(model, device_ids=self.device_ids)
        #                .to(device))
        self.model = DataParallel(model).to(device)
        self.opt = None

    def construct_optimizer(self, opt, *args, **kwargs):
        self.opt = opt(self.model.parameters(), *args, **kwargs)

    def train(self, *args, **kwargs):
        assert(self.opt is not None)
        self.opt.zero_grad()
        loss, samples = self.model(*args, **kwargs)
        loss = loss.mean()
        loss.backward()
        self.opt.step()
        return (loss.item(), samples)

    def eval(self, *args, **kwargs):
        with torch.no_grad():
            loss, samples = self.model(*args, **kwargs)
            loss = loss.mean()
            return (loss.item(), samples)
