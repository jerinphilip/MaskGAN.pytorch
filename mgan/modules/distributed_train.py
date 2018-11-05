from .distributed_model import DistributedModel

class DistributedTrain:
    def __init__(self, model, opt):
        assert(isinstance(model, DistributedModel))
        self.model = model
        self.distributed_model = DistributedDataParallel(model)
        self.opt = opt

    def train(self, inputs, targets):
        self.opt.zero_grad()
        loss = self.distributed_model(inputs, targets).mean()
        loss.backward()
        self.opt.step()
        return loss.item()

    def eval(self, inputs, targets):
        with torch.no_grad():
            loss = self.model(inputs, targets).mean()
            return loss.item()
