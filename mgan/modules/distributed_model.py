from torch import nn
from mgan.criterions import TCELoss, REINFORCE
from mgan.models import MLEGenerator, MLEGenerator, MGANDiscriminator

class DistributedModel(nn.Module):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion

    def forward(*args, **kwargs):
        raise NotImplementedError



class MLEDistributedGenerator(DistributedModel):
    @classmethod
    def build(cls, args, task):
        model = MLEGenerator.build(args, task)
        criterion = TCELoss()
        return cls(model, criterion)

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        net_output = self.model(src_tokens, 
                src_lengths, prev_output_tokens)

        logits = net_output[0].float()
        logits = logits[:, :-1, :].contiguous()
        target = prev_output_tokens[:, 1:].contiguous().view(-1)

        loss = self.criterion(logits, target)
        return loss

class MGANGDistributedGenerator(DistributedModel):
    @classmethod
    def build(cls, args, task):
        model = MLEGenerator.build(args, task)
        criterion = REINFORCE()
        return cls(model, criterion)

    def forward(self, src_tokens, src_lengths, prev_output_tokens, discriminator):
        samples, log_probs, attns = self.model(src_tokens, 
                        src_lengths, prev_output_tokens)
        logits, attn_scores = discriminator(samples, 
                src_lengths, prev_output_tokens)
        reward = self.criterion(log_probs, logits)
        loss = -1*reward.mean()
        return (loss, samples)


class MGANGDistributedDiscriminator(DistributedModel):
    @classmethod
    def build(cls, args, task):
        model = MGANGDiscriminator.build(args, task)
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
        return loss



