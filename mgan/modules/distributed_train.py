from .distributed_model import DistributedModel

class DistributedTrain:
    def __init__(self, model, opt):
        assert(isinstance(model, DistributedModel))
        self.model = model
        self.distributed_model = DistributedDataParallel(model)
        self.opt = opt

    def train(self, *args):
        self.opt.zero_grad()
        loss = self.distributed_model(*args).mean()
        loss.backward()
        self.opt.step()
        return loss.item()

    def eval(self, *args):
        with torch.no_grad():
            loss = self.model(*args).mean()
            return loss.item()
