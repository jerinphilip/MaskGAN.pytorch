
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

class MGANGEncoder(LSTMEncoder): pass
class MGANGDecoder(LSTMDecoder): pass
class MGANGenerator(LSTMModel):
    def forward(self, src_tokens, src_lengths, prev_output_tokens, src_mask):
        self.encoder.lstm.flatten_parameters()
        logits, attns = super().forward(src_tokens, src_lengths, prev_output_tokens)
        bsz, seqlen, vocab_size = logits.size()
        # print("Logits size:", logits.size())

        # Sample from x converting it to probabilities
        samples = []
        log_probs = []
        for t in range(seqlen):
            # input is B x T x C post transposing
            logit = logits[:, t, :]
            # Good news, categorical works for a batch.
            # B x H dimension. Looks like logit's are already in that form.
            EPS = 1e-7
            logit = logit + EPS
            distribution = Categorical(logits=logit)
            # Output is H dimension?
            sampled = distribution.sample()
            fsampled = torch.where(src_mask[:, t].byte(), sampled, prev_output_tokens[:, t])
            log_probs.append(distribution.log_prob(fsampled))
            samples.append(fsampled)
            

        # Once all are sampled, it's possible to find the rewards from the generator.
        samples = torch.stack(samples, dim=1)
        log_probs = torch.stack(log_probs, dim=1)
        # I may need to strip off an extra token generated.
        # Nope, I may not need to. Yes, I may need to, at the end though.
        samples = samples[:, 1:]
        return (samples, log_probs, attns)

class MLEGenerator(LSTMModel):
    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        self.encoder.lstm.flatten_parameters()
        return super().forward(src_tokens, src_lengths, prev_output_tokens)

