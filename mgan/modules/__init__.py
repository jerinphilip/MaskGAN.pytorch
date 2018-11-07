
from .distributed_model import DistributedModel
from .distributed_train import DistributedTrain

from .distributed_model import          \
        MGANDistributedGenerator,      \
        MGANDistributedDiscriminator,  \
        MLEDistributedGenerator

import torch


class MGANTrainer:
    def __init__(self, args, task):
        generator = MGANDistributedGenerator.build_model(args, task)
        discriminator = MGANDistributedDiscriminator.build_model(args, task)

        gopt = torch.optim.Adam(generator.parameters())
        self.generator = DistributedTrain(generator, gopt)

        dopt = torch.optim.Adam(discriminator.parameters())
        self.discriminator = DistributedTrain(discriminator, dopt)

        self.savable = [
            ("mgan-generator", self.generator),
            ("mgan-discriminator", self.discriminator)
        ]

    def __call__(self, src_tokens, src_lengths, src_mask, 
            tgt_tokens, tgt_lengths, tgt_mask):

        prev_output_tokens = tgt_tokens

        g_steps, d_steps = 10, 10

        gloss, d_real_loss, d_fake_loss = 0, 0, 0

        for step in range(g_steps):
            _gloss, samples = self.generator(src_tokens, src_lengths, 
                    prev_output_tokens, self.discriminator.model.model)
            gloss += _gloss


        for step in range(d_steps):
            _d_real_loss, _ = self.discriminator(
                            prev_output_tokens[:, 1:], src_lengths, 
                            prev_output_tokens, real=True)


            _gloss, samples = self.generator.eval(src_tokens, src_lengths, 
                    prev_output_tokens, self.discriminator.model.model)

            _d_fake_loss, _  = self.discriminator(
                             samples, src_lengths, 
                             prev_output_tokens, real=False)
            
            d_real_loss += _d_real_loss
            d_fake_loss += _d_fake_loss


        return {
                "Generator Loss": gloss/g_steps,
                "Discriminator Real Loss": d_real_loss/d_steps,
                "Discriminator Fake Loss": d_fake_loss/d_steps
        }



class MLETrainer:
    def __init__(self, args, task):
        generator = MLEDistributedGenerator.build_model(args, task)

        gopt = torch.optim.Adam(generator.parameters())
        self.generator = DistributedTrain(generator, gopt)

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


