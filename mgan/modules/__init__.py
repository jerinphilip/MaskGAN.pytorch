
from .distributed_train import DistributedTrain

from .distributed_model import          \
        MLEDistributedGenerator, MGANModel
from torch.nn.parallel import DataParallel

import torch


class MGANTrainer:
    def __init__(self, args, task):
        model = MGANModel.build_model(args, task)
        device = torch.device("cuda")
        self.model = DataParallel(model)
        self.opt = torch.optim.Adam(self.model.parameters())
        self.model = self.model.to(device)

        self.dopt = self.opt
        self.gopt = self.opt


    def run(self, src_tokens, src_lengths, src_mask, 
            tgt_tokens, tgt_lengths, tgt_mask):

        prev_output_tokens = tgt_tokens
        g_steps, d_steps = 10, 10
        gloss, d_real_loss, d_fake_loss = 0, 0, 0


        for step in range(d_steps):
            self.dopt.zero_grad()

            _d_real_loss, _ = self.model("d-step",
                            prev_output_tokens[:, 1:], src_lengths, 
                            prev_output_tokens, real=True)

            with torch.no_grad():
                _gloss, samples = self.model("g-step", 
                                src_tokens, src_lengths, 
                                prev_output_tokens)

            _d_fake_loss, _  = self.model("d-step",
                             samples, src_lengths, 
                             prev_output_tokens, real=False)
            loss = (_d_real_loss + _d_fake_loss)/2
            loss.backward()
            self.dopt.step()
            
            d_real_loss += _d_real_loss.item()
            d_fake_loss += _d_fake_loss.item()

        for step in range(g_steps):
            self.gopt.zero_grad()
            _gloss, samples = self.model("g-step", src_tokens, src_lengths, 
                    prev_output_tokens)
            _gloss.backward()
            self.gopt.step()
            gloss += _gloss.item()


        return {
                "Generator Loss": gloss/g_steps,
                "Discriminator Real Loss": d_real_loss/d_steps,
                "Discriminator Fake Loss": d_fake_loss/d_steps
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


