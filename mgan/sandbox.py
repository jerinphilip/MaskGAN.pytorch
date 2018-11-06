# General Imports
from argparse import ArgumentParser
from tqdm import tqdm
from collections import namedtuple

# Torch imports
import torch
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader


# FairSeq imports
from fairseq.meters import AverageMeter
from fairseq.progress_bar import tqdm_progress_bar
from fairseq.sequence_generator import SequenceGenerator

from mgan.preproc import Preprocess
from mgan.data import IMDbDataset, TensorIMDbDataset
# from mgan.models import MaskGAN
# from mgan.models import train, pretrain
from mgan.modules import build_trainer
from mgan.utils import Saver
from mgan.report_hooks import visdom


class Args: 
    criterion = 'dummy'

def dataset_test(args):
    mask = {
        "type": "random",
        "kwargs": {"probability": 0.1}
    }

    tokenize = {
        "type": "space",
    }

    preprocess = Preprocess(mask, tokenize)
    dataset = TensorIMDbDataset(args.path, preprocess, truncate=20)
    loader = DataLoader(dataset, batch_size=160, collate_fn=TensorIMDbDataset.collate, shuffle=True, num_workers=16)
    Task = namedtuple('Task', 'source_dictionary target_dictionary')
    task = Task(source_dictionary=dataset.vocab, target_dictionary=dataset.vocab)

    meters = {}
    meters['epoch'] = AverageMeter()
    meters['loss'] = AverageMeter()

    device = torch.device('cuda')

    args = Args()
    max_epochs = 100

    checkpoint_path = "/scratch/jerin/mgan/"
    saver = Saver(checkpoint_path)
    trainer = build_trainer("MLE", args, task)

    saver.load(trainer.generator.model, trainer.generator.opt, "generator-pretrain")

    for epoch in tqdm(range(max_epochs), total=max_epochs, desc='epoch'):
        pbar = tqdm_progress_bar(loader, epoch=epoch)
        meters["loss"].reset()
        count = 0
        for src, src_lens, src_mask, tgt, tgt_lens, tgt_mask in pbar:
            count += 1
            src, tgt = src.to(device), tgt.to(device)
            summary = trainer(src, src_lens, src_mask, tgt, tgt_lens, tgt_mask)
            visdom.log('generator-loss-vs-epoch', 'line', summary['Generator Loss'])

        avg_loss = meters["loss"].avg
        meters['epoch'].update(avg_loss)
        saver.checkpoint(trainer.generator.model, trainer.generator.opt, "generator-pretrain")

    # seq_gen = SequenceGenerator([model], dataset.vocab, beam_size=5)
    # for src, src_lens, tgt, tgt_lens in loader:
    #     src = src.to(device)
    #     encoder_input = {"src_tokens": src, "src_lengths": src_lens}
    #     samples = seq_gen.generate(encoder_input, maxlen=20)
    #     for i, sample in enumerate(samples):
    #        # print(sample[0].keys())
    #        src_str = dataset.vocab.string(src[i, :])
    #        tgt_str = dataset.vocab.string(tgt[i, :])
    #        pred_str = dataset.vocab.string(sample[0]['tokens'])
    #        print(">", src_str)
    #        print("<", pred_str)
    #        print("=", tgt_str)
    #        print("")
    #     break



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', required=True)
    args = parser.parse_args()
    dataset_test(args)

