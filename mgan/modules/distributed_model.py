import torch
from torch import nn
from mgan.criterions import TCELoss, REINFORCE
from mgan.models import MLEGenerator, MLEGenerator, \
        MGANDiscriminator, MGANGenerator

class DistributedModel(nn.Module):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion
        print(self.model, self.criterion)

    def load_state_dict(self, state):
        self.model.load_state_dict(state)

    def state_dict(self):
        return self.model.state_dict()

    def forward(self, *args, **kwargs):
        raise NotImplementedError



class MLEDistributedGenerator(DistributedModel):
    @classmethod
    def build_model(cls, args, task):
        model = MLEGenerator.build_model(args, task)
        criterion = TCELoss()
        return cls(model, criterion)

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        net_output = self.model(src_tokens, 
                src_lengths, prev_output_tokens)

        logits = net_output[0].float()
        logits = logits[:, :-1, :].contiguous()
        target = prev_output_tokens[:, 1:].contiguous().view(-1)

        loss = self.criterion(logits, target)
        return (loss, None)

class MGANDistributedGenerator(DistributedModel):
    @classmethod
    def build_model(cls, args, task):
        model = MGANGenerator.build_model(args, task)
        criterion = REINFORCE(gamma=0.6)
        return cls(model, criterion)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, discriminator):
        samples, log_probs, attns = self.model(src_tokens, 
                        src_lengths, prev_output_tokens)
        logits, attn_scores = discriminator(samples, 
                src_lengths, prev_output_tokens)
        reward = self.criterion(log_probs, logits)
        loss = -1*reward.mean()
        return (loss, samples)


class MGANDistributedDiscriminator(DistributedModel):
    @classmethod
    def build_model(cls, args, task):
        model = MGANDiscriminator.build_model(args, task)
        criterion = torch.nn.BCEWithLogitsLoss()
        return cls(model, criterion)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, real=True):
        logits, attn_scores = self.model(
                prev_output_tokens[:, 1:], 
                src_lengths, 
                prev_output_tokens)

        truths = torch.ones_like(logits)
        if not real:
            truths = torch.zeros_like(logits)

        loss = self.criterion(logits,  truths)
        return (loss, None)



