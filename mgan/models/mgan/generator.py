
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

class MGANGEncoder(LSTMEncoder):
    pass

class MGANGDecoder(LSTMDecoder):

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, prev_output_tokens, encoder_out_dict, incremental_state=None):
        x, attn_scores = super().forward(prev_output_tokens, encoder_out_dict, incremental_state)
        bsz, seqlen = prev_output_tokens.size()
        # Sample from x converting it to probabilities
        samples = []
        for t in range(seqlen):
            # input is B x T x C post transposing
            logits = x[:, t, :]
            # Good news, categorical works for a batch.
            # B x H dimension. Looks like logit's are already in that form.
            distribution = Categorical(logits=logits)

            # Output is H dimension?
            sampled = distributions.sample()
            samples.append(sampled)

        # Concatenenate in a new dimension
        sampled = torch.cat(samples)
        """
    pass


