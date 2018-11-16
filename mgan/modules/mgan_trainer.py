import torch
from torch.nn.parallel import DataParallel
from .distributed_model import MGANModel


class MGANTrainer:
    def __init__(self, args, task):
        device = torch.device("cuda")
        self.pretrain = True
        self._model = MGANModel.build_model(args, task, pretrain=self.pretrain)
        self.model = DataParallel(self._model)
        self.model = self.model.to(device)
        self.opt = torch.optim.Adam(self.model.parameters())

        self.savable = [
             ["mgan-model", self.model.module]
        ]


    def run(self, epoch, samples):
        g_steps, d_steps = 5, 5
        summary = {}
        discriminator_summary = self.run_dsteps(d_steps, samples)
        torch.cuda.empty_cache()
        generator_summary = self.run_gsteps(g_steps, samples)
        torch.cuda.empty_cache()

        summary.update(discriminator_summary)
        summary.update(generator_summary)
        return summary

    def run_dsteps(self, d_steps, samples):
        src_tokens, src_lengths, src_mask, \
            tgt_tokens, tgt_lengths, tgt_mask = samples

        prev_output_tokens = tgt_tokens
        d_real_loss, d_fake_loss = 0, 0,
        for step in range(d_steps):
            self.opt.zero_grad()
            _d_real_loss, _ = self.model(prev_output_tokens[:, 1:], 
                    src_lengths, tgt_mask, prev_output_tokens, 
                    tag="d-step", real=True)
            _d_real_loss = _d_real_loss.mean()

            if self.pretrain:
                samples = src_tokens

            else:
                with torch.no_grad():
                    _gloss, samples, _closs = self.model(src_tokens, src_lengths, src_mask,
                                    prev_output_tokens, tag="g-step")

            _d_fake_loss, _  = self.model(samples, src_lengths, tgt_mask,
                             prev_output_tokens, 
                             tag="d-step", real=False)
            _d_fake_loss = _d_fake_loss.mean()

            loss = (_d_real_loss + _d_fake_loss )/2
            loss.backward()
            
            d_real_loss += _d_real_loss.item()
            d_fake_loss += _d_fake_loss.item()
            self.opt.step()


        return {
                "Discriminator Real Loss": d_real_loss/d_steps,
                "Discriminator Fake Loss": d_fake_loss/d_steps,
                "Discriminator Overall Loss": (d_fake_loss+d_real_loss)/(2*d_steps)
        }

    def run_gsteps(self, g_steps, samples):
        src_tokens, src_lengths, src_mask, \
            tgt_tokens, tgt_lengths, tgt_mask = samples

        prev_output_tokens = tgt_tokens
        gloss = 0
        closs = 0

        for step in range(g_steps):
            self.opt.zero_grad()
            _gloss, samples, _closs = self.model(src_tokens, src_lengths, src_mask,
                    prev_output_tokens, tag="g-step")

            _gloss = _gloss.mean()
            _gloss.backward()
            gloss += _gloss.item()

            if not self.pretrain:
                _closs = _closs.mean()
                _closs.backward()
                closs += _closs.item()

            self.opt.step()


        return {
                "Generator Loss": gloss/g_steps,
                "Critic Loss": closs/g_steps
        }



