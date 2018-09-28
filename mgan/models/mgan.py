
from fairseq.models.lstm \
        import LSTMEncoder, \
               LSTMDecoder, \
               LSTMModel

from fairseq.models.fairseq_model \
        import FairseqModel

from torch.distributions.categorical import Categorical
from warnings import warn
from torch import nn

class MaskedGAN(nn.Module):
    """
    MaskedGAN doesn't obey FairseqModel's rules.
    """
    def __init__(self, generator, discriminator, critic):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.critic = critic

    @classmethod
    def build_model(cls, args):
        pass

    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        pass


class MGANDecoder(LSTMDecoder):

    def sample_token(self, net_outputs):
        """
        Build's a categorical distribution over the probabilities output by the
        network. Samples a token.

        """

        warn("Check logit dimensions checkout")
        logits = net_output[0].float()

        warn("Categorical works?")
        dist = Categorical(logits)
        idx = dist.sample()
        pdb.set_trace()


    def forward(self, prev_output_tokens, encoder_out_dict, incremental_state=None):
        encoder_out = encoder_out_dict['encoder_out']
        encoder_padding_mask = encoder_out_dict['encoder_padding_mask']

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, _, _ = encoder_out[:3]
        srclen = encoder_outs.size(0)

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            prev_hiddens, prev_cells, input_feed = cached_state
        else:
            _, encoder_hiddens, encoder_cells = encoder_out[:3]
            num_layers = len(self.layers)
            prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
            prev_cells = [encoder_cells[i] for i in range(num_layers)]
            input_feed = x.data.new(bsz, self.encoder_output_units).zero_()

        attn_scores = x.data.new(srclen, seqlen, bsz).zero_()
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j, :, :], input_feed), dim=1)

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
                out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs, encoder_padding_mask)
            else:
                out = hidden
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # input feeding
            input_feed = out

            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state', (prev_hiddens, prev_cells, input_feed))

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn:
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None

        # project back to size of vocabulary
        if self.adaptive_softmax is None:
            if hasattr(self, 'additional_fc'):
                x = self.additional_fc(x)
                x = F.dropout(x, p=self.dropout_out, training=self.training)
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)
        return x, attn_scores



    



class MaskedRL(LSTMModel):
    def forward(self, src_tokens, src_lengths, prev_output_tokens):
        """
        Instead of taking argmax, use an RL agent to sample actions leading to
        the final sequence.

        Will need to return logprobs, I think.

        """
        pass


