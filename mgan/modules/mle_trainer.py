from .distributed_train import DistributedTrain

from .distributed_model import          \
        MLEDistributedGenerator, MGANModel
from torch.nn.parallel import DataParallel
import torch

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



