import torch
from torch import nn
from collections import namedtuple

from mgan.criterions import \
        TCELoss,            \
        REINFORCE,          \
        TBCELoss,           \
        WeightedMSELoss

from mgan.models import     \
        MLEGenerator,       \
        MGANDiscriminator,  \
        MGANGenerator,      \
        MGANCritic



class LossModel(nn.Module):
    def __init__(self, model, criterion):
        super().__init__()
        self.model = model
        self.criterion = criterion


class MGANModel(nn.Module):
    def __init__(self, generator, discriminator, critic=None, pretrain=False):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.critic = critic
        self.pretrain = pretrain

    @classmethod
    def build_model(cls, args, task, pretrain):
        # Build critic
        critic = MGANCritic.build_model(args, task)
        mse_loss = WeightedMSELoss()
        closs = LossModel(critic, mse_loss)

        # Build generator
        if pretrain:
            generator = MLEGenerator.build_model(args,task)
            gcriterion = TCELoss()
        else:
            generator = MGANGenerator.build_model(args, task)
            reinforce = REINFORCE(gamma=0.6, clip_value=1)
            gcriterion = reinforce

        gloss = LossModel(generator, gcriterion)

        # Build discriminator
        discriminator = MGANDiscriminator.build_model(args, task)
        tceloss = TBCELoss()
        dloss = LossModel(discriminator, tceloss)

        return cls(gloss, dloss, closs, pretrain=pretrain)

    def forward(self, *args, **kwargs):
        if kwargs['tag'] == 'g-step':
            if self.pretrain:
                return self._gstep_pretrain(*args)
            else:
                return self._gstep(*args)
        return self._dstep(*args, real=kwargs['real'])

    def _gstep(self, src_tokens, src_lengths, src_mask, prev_output_tokens):
        samples, log_probs, attns = self.generator.model(src_tokens, 
                        src_lengths, prev_output_tokens)

        with torch.no_grad():
            logits, attn_scores = self.discriminator.model(samples, 
                    src_lengths, prev_output_tokens)

        baselines, _ = self.critic.model(samples, src_lengths, prev_output_tokens)
        reward, cumulative_rewards = self.generator.criterion(log_probs, 
                logits, src_mask, baselines.detach())

        gloss = -1*reward
        critic_loss = self.critic.criterion(baselines.squeeze(2), 
                cumulative_rewards.detach(), src_mask)

        return (gloss, samples, critic_loss)

    def _gstep_pretrain(self, src_tokens, src_lengths, 
            src_mask, prev_output_tokens):
        logits, attns = self.generator.model(src_tokens, 
                        src_lengths, prev_output_tokens)

        loss = self.generator.criterion(logits, prev_output_tokens)
        return (loss, None, None)

    def _dstep(self, src_tokens, src_lengths, 
            src_mask, prev_output_tokens, real=True):
        logits, attn_scores = self.discriminator.model(src_tokens, 
                src_lengths, prev_output_tokens)
        src_mask = src_mask.unsqueeze(2)
        truths = torch.ones_like(logits) if real \
                else torch.ones_like(logits) - src_mask[:, :-1]

        loss = self.discriminator.criterion(logits, truths, weight=src_mask[:, :-1])
        return (loss, None)


