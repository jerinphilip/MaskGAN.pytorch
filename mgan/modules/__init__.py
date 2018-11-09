
from .distributed_train import DistributedTrain

from .distributed_model import          \
        MLEDistributedGenerator, MGANModel
from torch.nn.parallel import DataParallel

import torch


class MGANTrainer:
    def __init__(self, args, task):
        model = MGANModel.build_model(args, task)
        self.model = DataParallel(model)
        device = torch.device("cuda")
        self.model = self.model.to(device)
        self.opt = torch.optim.SGD(self.model.parameters(), lr=0.01)

        self.dopt = self.opt
        self.gopt = self.opt


    def run(self, *args):
        g_steps, d_steps = 10, 10
        summary = {}
        discriminator_summary = self.run_dsteps(d_steps, *args)
        generator_summary = self.run_gsteps(g_steps, *args)

        summary.update(discriminator_summary)
        summary.update(generator_summary)
        return summary

    def run_dsteps(self, d_steps, src_tokens, src_lengths, src_mask, 
            tgt_tokens, tgt_lengths, tgt_mask):

        prev_output_tokens = tgt_tokens
        d_real_loss, d_fake_loss = 0, 0,
        for step in range(d_steps):
            self.dopt.zero_grad()
            _d_real_loss, _ = self.model(prev_output_tokens[:, 1:], src_lengths, 
                            prev_output_tokens, tag="d-step", real=True)

            _d_real_loss = _d_real_loss.mean()

            with torch.no_grad():
                _gloss, samples = self.model(src_tokens, src_lengths, 
                                prev_output_tokens, tag="g-step")

            _d_fake_loss, _  = self.model(samples, src_lengths, 
                             prev_output_tokens, tag="d-step", real=False)

            _d_fake_loss = _d_fake_loss.mean()
            loss = (_d_real_loss + _d_fake_loss)/2
            loss.backward()
            self.dopt.step()
            
            d_real_loss += _d_real_loss.item()
            d_fake_loss += _d_fake_loss.item()

        return {
                "Discriminator Real Loss": d_real_loss/d_steps,
                "Discriminator Fake Loss": d_fake_loss/d_steps
        }

    def run_gsteps(self, g_steps, src_tokens, src_lengths, src_mask, 
            tgt_tokens, tgt_lengths, tgt_mask):

        prev_output_tokens = tgt_tokens
        gloss = 0

        for step in range(g_steps):
            self.gopt.zero_grad()
            _gloss, samples = self.model(src_tokens, src_lengths, 
                    prev_output_tokens, tag="g-step")
            _gloss = _gloss.mean()
            _gloss.backward()
            self.gopt.step()
            gloss += _gloss.item()


        return {
                "Generator Loss": gloss/g_steps
        }



class MLETrainer:
    def __init__(self, args, task):
        generator = MLEDistributedGenerator.build_model(args, task)

        self.generator = DistributedTrain(generator)
        self.generator.construct_optimizer(torch.optim.Adam)

        self.savable = [
            ("mle-generator", self.generator),
        ]

    def __call__(self, src_tokens, src_lengths, src_mask,
            tgt_tokens, tgt_lengths, tgt_mask):

        gloss, _ = self.generator(src_tokens, src_lengths, tgt_tokens)
        return {"Generator Loss": gloss}



def build_trainer(tag, args, task):
    if tag == 'MLE':
        trainer = MLETrainer(args, task)
        return trainer

    elif tag == 'MGAN':
        trainer = MGANTrainer(args, task)
        return trainer
    
    else:
        raise Exception("Unknown tag")


