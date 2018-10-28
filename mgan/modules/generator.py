
import torch.nn.functional as F
from torch import nn
import torch

class Generator(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args):
        net_output = self.model(*args)
        logits = net_output[0].float()
        return F.log_softmax(logits, dim=-1)

class LossGenerator(nn.Module):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion


    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        net_output = self.model(src_tokens, src_lengths, prev_output_tokens) 
        logits = net_output[0].float()
        lprobs = F.log_softmax(logits, dim=-1)

        # B x T x H sequence
        lprobs = lprobs[:, :-1, :].contiguous()
        lprobs = lprobs.view(-1, lprobs.size(-1))

        target = prev_output_tokens[:, 1:].contiguous().view(-1)
        loss = self.criterion(lprobs, target)
        return loss



class LossModel(nn.Module):
    def __init__(self, model, criterion):
        self.model = model
        self.criterion = criterion

    def forward(*args, **kwargs):
        raise NotImplementedError


class DistributedTrain:
    def __init__(self, model, opt):
        assert(isinstance(model, LossModel))
        self.model = model
        self.distributed_model = DistributedDataParallel(model)
        self.opt = opt

    def train(self, inputs, targets):
        loss = self.distributed_model(inputs, targets).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def eval(self, inputs, targets):
        with torch.no_grad():
            loss = self.model(inputs, targets).mean()
            return loss.item()
