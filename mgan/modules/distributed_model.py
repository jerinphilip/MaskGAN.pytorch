import torch
from torch import nn
from mgan.criterions import TCELoss, REINFORCE
from mgan.models import MLEGenerator, MLEGenerator, \
        MGANDiscriminator, MGANGenerator

from collections import namedtuple


class LossModel(nn.Module):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion


class MGANModel(nn.Module):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    @classmethod
    def build_model(cls, args, task):
        # Build generator
        generator = MGANGenerator.build_model(args, task)
        reinforce = REINFORCE(gamma=0.6)
        gloss = LossModel(generator, reinforce)

        # Build discriminator
        discriminator = MGANDiscriminator.build_model(args, task)
        #bceloss = torch.nn.BCEWithLogitsLoss()
        tceloss = TCELoss()
        dloss = LossModel(discriminator, tceloss)

        return cls(gloss, dloss)

    def forward(self, *args, **kwargs):
        if kwargs['tag'] == 'g-step':
            return self._gstep(*args)
        return self._dstep(*args, real=kwargs['real'])

    def _gstep(self, src_tokens, src_lengths, src_mask, prev_output_tokens):

        samples, log_probs, attns = self.generator.model(src_tokens, 
                        src_lengths, prev_output_tokens)

        logits, attn_scores = self.discriminator.model(samples, 
                src_lengths, prev_output_tokens)

        samples = samples.detach()
        logits = logits.detach()

        reward = self.generator.criterion(log_probs, logits)
        loss = -1*reward
        return (loss, samples)

    def _dstep(self, src_tokens, src_lengths, src_mask, prev_output_tokens, real=True):
        logits, attn_scores = self.discriminator.model(
                prev_output_tokens[:, 1:], 
                src_lengths, 
                prev_output_tokens)

        src_mask = src_mask.unsqueeze(2)
        truths = torch.ones_like(logits) if real else torch.ones_like(logits) - src_mask

        loss = self.discriminator.criterion(logits, truths, weight=src_mask)
        return (loss, None)


class MLEDistributedGenerator(nn.Module):
    def __init__(self, model, criterion):
        super().__init__()
        self._lm = LossModel(model, criterion)

    @classmethod
    def build_model(cls, args, task): 
        model = MLEGenerator.build_model(args, task)
        criterion = TCELoss()
        return cls(model, criterion)

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        net_output = self._lm.model(src_tokens, 
                src_lengths, prev_output_tokens)

        logits = net_output[0].float()
        logits = logits[:, :-1, :].contiguous()
        target = prev_output_tokens[:, 1:].contiguous().view(-1)

        loss = self._lm.criterion(logits, target)
        return (loss, None)
