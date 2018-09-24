from torch import nn
import torch
from mgan.report_hooks import DevNull

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, vocab, 
            max_length=30, teacher_forcing=0.5, report_hook=DevNull()):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.max_length = max_length
        self.teacher_forcing = teacher_forcing
        self.vocab = vocab
        self.report_hook = report_hook

    def forward(self, x, lx, 
            z = None, lz = None):
        e_out, e_hidden = self.encoder(x, lx)
        if z is not None:
            return self._teacher_forced_decoder_forward(e_hidden, e_out, z, lz) 
        else:
            return self._naive_decoder_unroll(e_hidden, e_out)
    
    def _teacher_forced_decoder_forward(self, 
            d_hidden, e_out, z, lz):
        d_ctx = d_hidden
        d_outs = []
        T, B = z.size()
        for t in range(T):
            seed = z[t, :]
            d_out, d_hidden, d_ctx, attns = self.decoder(
                    seed, d_hidden, d_ctx, e_out
            )
            
            self.report_hook({
                "t": t,
                "out": d_out.detach(), 
                "hidden": d_hidden.detach(),
                "attns": attns.detach()
            })
            
            d_outs.append(d_out)

            
        # Output is a TxBxH
        # Concatenate on dimension 0
        # TODO: Possiblity of a leak. Needs to be handled.
        return torch.cat(d_outs, 0)
    

    def _naive_decoder_unroll(self, d_hidden, e_out):
        d_ctx = d_hidden
        d_outs = []
        device = d_hidden.device
        bos = self.vocab.special_idxs.bos 
        seed = torch.LongTensor([bos]).to(device)
        T, B, H = e_out.size()
        seed = seed.repeat(B).view(B)
        print("Seed size:", seed.size())
        for t in range(self.max_length):
            d_out, d_hidden, d_ctx, attns = self.decoder(
                    seed, d_hidden, d_ctx, e_out
            )

            self.report_hook({
                "t": t,
                "out": d_out.detach(), 
                "hidden": d_hidden.detach(),
                "attns": attns.detach()
            })

            d_outs.append(d_out)
            mv, mi = d_out.max(dim=2)
            seed = mi.view(-1)

        return torch.cat(d_outs, 0)



