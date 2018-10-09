

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
from .critic import MGANCritic

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
        critic = MGANCritic.build_model(args, task)
        return cls(generator, discriminator, critic)

    """
    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        logits, attns = self.generator(src_tokens, src_lengths, prev_output_tokens)
        bsz, seqlen, vocab_size = logits.size()
        # print("Logits size:", logits.size())

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
        # I may need to strip off an extra token generated.
        samples = samples[:, 1:]
        warn("Samples may not be the correct size. May need fixing")
        # print("Samples:", samples.size(), "src_tokens_size:", src_tokens.size())
        probs, attn_scores = self.discriminator(samples, src_lengths, prev_output_tokens)
        r = []
        for t in range(seqlen-1):
            _r = torch.log(probs[:, t])
            r.append(_r)

        R = [0 for i in range(seqlen)]
        gamma_0 = 0.95
        gamma_t = gamma_0
        for t in reversed(range(seqlen-1)):
            R[t] = gamma_t * r[t] + R[t+1]
            gamma_t = gamma_t*gamma_0

        E_R = 0
        for t in range(seqlen-1):
            d = distribution[t]
            E_R += R[t]*d.log_prob(samples[:, t])

        return -1*E_R
    """



def train(model, opt, src_token, src_lengths, prev_output_tokens):
    logits, attns = model.generator(src_tokens, src_lengths, prev_output_tokens)
    bsz, seqlen, vocab_size = logits.size()
    # print("Logits size:", logits.size())

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
    # I may need to strip off an extra token generated.
    samples = samples[:, 1:]
    warn("Samples may not be the correct size. May need fixing")
    # print("Samples:", samples.size(), "src_tokens_size:", src_tokens.size())
    probs, attn_scores = model.discriminator(samples, src_lengths, prev_output_tokens)
    r = []
    for t in range(seqlen-1):
        _r = torch.log(probs[:, t])
        r.append(_r)

    R = [0 for i in range(seqlen)]
    gamma_0 = 0.95
    gamma_t = gamma_0
    for t in reversed(range(seqlen-1)):
        R[t] = gamma_t * r[t] + R[t+1]
        gamma_t = gamma_t*gamma_0

    E_R = 0
    for t in range(seqlen-1):
        d = distribution[t]
        E_R += R[t]*d.log_prob(samples[:, t])

    gloss = -1*E_R

    opt.zero_grad()
    gloss.backward()
    opt.step()


    # expected_rewards, attn_scores = model.critic(samples, src_lengths, prev_output_tokens)
    # loss = expected_rewards - torch.Tensor(


    # need to create a mix of real and fake data.

    dloss = 

