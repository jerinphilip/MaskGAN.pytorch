from .distributed_train import DistributedTrain

from .distributed_model import          \
        MLEDistributedGenerator, MGANModel
from torch.nn.parallel import DataParallel

import torch


class MGANTrainer:
    def __init__(self, args, task):
        device = torch.device("cuda")
        self._model = MGANModel.build_model(args, task)
        self.model = DataParallel(self._model)
        self.model = self.model.to(device)
        # self.opt = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.opt = torch.optim.Adam(self.model.parameters())
        self.dopt = self.opt
        self.gopt = self.opt

        self.savable = [
             ["mgan-model", self.model.module]
        ]


    def run(self, *args):
        g_steps, d_steps = 5, 5
        summary = {}
        discriminator_summary = self.run_dsteps(d_steps, *args)
        torch.cuda.empty_cache()
        generator_summary = self.run_gsteps(g_steps, *args)
        torch.cuda.empty_cache()
        # generator_summary = {"Generator Loss": 0}

        summary.update(discriminator_summary)
        summary.update(generator_summary)
        return summary

    def run_dsteps(self, d_steps, src_tokens, src_lengths, src_mask, 
            tgt_tokens, tgt_lengths, tgt_mask):

        prev_output_tokens = tgt_tokens
        d_real_loss, d_fake_loss = 0, 0,
        for step in range(d_steps):
            self.dopt.zero_grad()
            _d_real_loss = torch.Tensor([0])
            _d_real_loss, _ = self.model(prev_output_tokens[:, 1:], src_lengths, tgt_mask,
                            prev_output_tokens, tag="d-step", real=True)


            _d_real_loss = _d_real_loss.mean()
            # _d_real_loss.backward()
            # self.dopt.step()

            # self.dopt.zero_grad()
            with torch.no_grad():
                _gloss, samples = self.model(src_tokens, src_lengths, src_mask,
                                prev_output_tokens, tag="g-step")
            # print(_gloss)

            if step == 0 and False:
                print("Samples", samples[0, :])
                print("SRCS", src_tokens[0, :])
                print("Targets", prev_output_tokens[0, 1:])

            _d_fake_loss, _  = self.model(src_tokens, src_lengths, tgt_mask,
                             prev_output_tokens, tag="d-step", real=False)
            # _d_fake_loss, _  = self.model(samples, src_lengths, src_mask,
            #                  prev_output_tokens, tag="d-step", real=False)
            # print(_d_fake_loss)

            _d_fake_loss = _d_fake_loss.mean()
            # _d_real_loss = _d_fake_loss
            # _d_fake_loss.backward()

            loss = (_d_real_loss + _d_fake_loss )/2
            loss.backward()
            
            d_real_loss += _d_real_loss.item()
            d_fake_loss += _d_fake_loss.item()
            self.dopt.step()


        return {
                "Discriminator Real Loss": d_real_loss/d_steps,
                "Discriminator Fake Loss": d_fake_loss/d_steps,
                "Discriminator Overall Loss": (d_fake_loss+d_real_loss)/(2*d_steps)
        }

    def run_gsteps(self, g_steps, src_tokens, src_lengths, src_mask, 
            tgt_tokens, tgt_lengths, tgt_mask):

        prev_output_tokens = tgt_tokens
        gloss = 0

        for step in range(g_steps):
            self.gopt.zero_grad()
            _gloss, samples = self.model(src_tokens, src_lengths, src_mask,
                    prev_output_tokens, tag="g-step")
            _gloss = _gloss.mean()
            _gloss.backward()
            self.gopt.step()
            gloss += _gloss.item()


        return {
                "Generator Loss": gloss/g_steps
        }



