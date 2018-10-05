

from fairseq.models.lstm \
        import LSTMEncoder, \
               LSTMDecoder, \
               LSTMModel, \
               base_architecture

from torch import nn
import torch

class MGANDiscriminatorEncoder(LSTMEncoder):
    pass

class MGANDiscriminatorDecoder(LSTMDecoder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.additional_fc = nn.Linear(self.hidden_size, 1)

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
        # Avoid this section. eee
        # if self.adaptive_softmax is None:
        #     if hasattr(self, 'additional_fc'):
        #         x = self.additional_fc(x)
        #         x = F.dropout(x, p=self.dropout_out, training=self.training)
        #     if self.share_input_output_embed:
        #         x = F.linear(x, self.embed_tokens.weight)
        #     else:
        #         x = self.fc_out(x)
        # return x, attn_scores

        x = self.additional_fc(x)
        x = torch.sigmoid(x)
        return (x, attn_scores)


class MGANDiscriminator(LSTMModel):
    @classmethod
    def build_model(cls, task, args):
       """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )

        if args.share_all_embeddings:
            # double check all parameters combinations are valid
            if task.source_dictionary != task.target_dictionary:
                raise RuntimeError('--share-all-embeddings requires a joint dictionary')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise RuntimeError(
                    '--share-all-embed not compatible with --decoder-embed-path'
                )
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise RuntimeError(
                    '--share-all-embeddings requires --encoder-embed-dim to '
                    'match --decoder-embed-dim'
                )
            pretrained_decoder_embed = pretrained_encoder_embed
            args.share_decoder_input_output_embed = True
        else:
            # separate decoder input embeddings
            pretrained_decoder_embed = None
            if args.decoder_embed_path:
                pretrained_decoder_embed = load_pretrained_embedding_from_file(
                    args.decoder_embed_path,
                    task.target_dictionary,
                    args.decoder_embed_dim
                )
        # one last double check of parameter combinations
        if args.share_decoder_input_output_embed and (
                args.decoder_embed_dim != args.decoder_out_embed_dim):
            raise RuntimeError(
                '--share-decoder-input-output-embeddings requires '
                '--decoder-embed-dim to match --decoder-out-embed-dim'
            )

        encoder = MGANDiscriminatorEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
        )
        decoder = MGANDiscriminatorDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.decoder_embed_dim,
            hidden_size=args.decoder_hidden_size,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=args.decoder_layers,
            dropout_in=args.decoder_dropout_in,
            dropout_out=args.decoder_dropout_out,
            attention=options.eval_bool(args.decoder_attention),
            encoder_embed_dim=args.encoder_embed_dim,
            encoder_output_units=encoder.output_units,
            pretrained_embed=pretrained_decoder_embed,
            share_input_output_embed=args.share_decoder_input_output_embed,
            adaptive_softmax_cutoff=(
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int)
                if args.criterion == 'adaptive_loss' else None
            ),
        )
        return cls(encoder, decoder)

