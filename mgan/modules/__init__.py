
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

    def __call__(self, src_tokens, src_lengths, src_mask, 
            tgt_tokens, tgt_lengths, tgt_mask):

        prev_output_tokens = tgt_tokens
        gloss, samples = self.generator(src_tokens, src_lengths, prev_output_tokens, self.discriminator.model.model)

        d_real_loss, _ = self.discriminator(
                        prev_output_tokens[:, 1:], src_lengths, 
                        prev_output_tokens, real=True)

        d_fake_loss, _  = self.discriminator(
                        samples, src_lengths, 
                        prev_output_tokens, real=False)

        return {
                "Generator Loss": gloss,
                "Discriminator Real Loss": d_real_loss,
                "Discriminator Fake Loss": d_fake_loss
        }



class MLETrainer:
    def __init__(self, args, task):
        generator = MLEDistributedGenerator.build_model(args, task)
        #discriminator = MGANGDistributedDiscriminator(args, task)

        gopt = torch.optim.Adam(generator.parameters())
        self.generator = DistributedTrain(generator, gopt)

        # dopt = torch.optim.Adam(discriminator.parameters())
        #self.discriminator = DistributedTrain(discriminator, dopt)
    def __call__(self, src_tokens, src_lengths, src_mask,
            tgt_tokens, tgt_lengths, tgt_mask):

        gloss = self.generator(src_tokens, src_lengths, tgt_tokens)
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


