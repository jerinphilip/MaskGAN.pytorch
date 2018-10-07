

from fairseq.models.lstm \
        import LSTMEncoder, \
               LSTMDecoder, \
               LSTMModel

from fairseq.models.fairseq_model \
        import FairseqModel

from torch.distributions.categorical import Categorical
from warnings import warn
from torch import nn
import torch

from .generator import MGANGenerator
from .discriminator import MGANDiscriminator

class MaskGAN(nn.Module):
    """
    MaskGAN doesn't obey FairseqModel's rules.
    """
    def __init__(self, generator, discriminator, critic):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.critic = critic

    @classmethod
    def build_model(cls, args, task):
        generator = MGANGenerator.build_model(args, task)
        discriminator = MGANDiscriminator.build_model(args, task)
        critic = None
        return cls(generator, discriminator, critic)

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        logits, attns = self.generator(src_tokens, src_lengths, prev_output_tokens)


        bsz, seqlen, vocab_size = logits.size()
        print("Logits size:", logits.size())

        # Sample from x converting it to probabilities
        samples = []
        distribution = {}
        for t in range(seqlen):
            # input is B x T x C post transposing
            logit = logits[:, t, :]
            # Good news, categorical works for a batch.
            # B x H dimension. Looks like logit's are already in that form.
            distribution[t] = Categorical(logits=logit)

            # Output is H dimension?
            sampled = distribution[t].sample().unsqueeze(1)
            samples.append(sampled)
            

        # Once all are sampled, it's possible to find the rewards from the generator.
        samples = torch.cat(samples, dim=1)
        print("Samples:", samples.size())
        probs = self.discriminator(samples, src_lengths, prev_output_tokens)
        rewards = []
        for t in range(seqlen):
            r = torch.log(probs[t])
            rewards.append(r)

