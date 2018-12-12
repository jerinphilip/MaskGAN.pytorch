
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
    def forward(self, masked, lengths, unmasked, mask):
        self.encoder.lstm.flatten_parameters()
        logits, attns = super().forward(masked, lengths, unmasked)
        bsz, seqlen, vocab_size = logits.size()

        # Sample from x converting it to probabilities
        samples = []
        log_probs = []
        for t in range(seqlen):
            logit = logits[:, t, :]
            distribution = Categorical(logits=logit)
            sampled = distribution.sample()
            fsampled = torch.where(mask[:, t].byte(), sampled, unmasked[:, t])
            log_prob = distribution.log_prob(fsampled)
            # flog_prob = torch.where(mask[:, t].byte(), log_prob, torch.zeros_like(log_prob))
            log_probs.append(log_prob)
            samples.append(fsampled)

        samples = torch.stack(samples, dim=1)
        log_probs = torch.stack(log_probs, dim=1)
        return (samples, log_probs, attns)

    def logits(self, masked, lengths, unmasked, mask):
        self.encoder.lstm.flatten_parameters()
        logits, attns = super().forward(masked, lengths, unmasked)
        return logits
    
class MLEGenerator(LSTMModel):
    def forward(self, masked, lengths, unmasked):
        self.encoder.lstm.flatten_parameters()
        return super().forward(masked, lengths, unmasked)

    def logits(self, masked, lengths, unmasked, mask):
        self.encoder.lstm.flatten_parameters()
        logits, attns = super().forward(masked, lengths, unmasked)
        return logits

