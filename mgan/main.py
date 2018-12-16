# General Imports
from argparse import ArgumentParser
from tqdm import tqdm
from collections import namedtuple
import gc
import os

# Torch imports
import torch
from torch import optim
from torch.utils.data import DataLoader
import random


# FairSeq imports
from fairseq.meters import AverageMeter
from fairseq.progress_bar import tqdm_progress_bar

from mgan.preproc import Preprocess, mask, tokenize
from mgan.data import IMDbDataset, TensorIMDbDataset
from mgan.modules import MGANTrainer
from mgan.utils import Saver
from mgan.utils.debug_generate import debug_generate
from mgan.utils.logging import visdom
from mgan.utils.leaks import leak_check, LeakCheck

def main(args):
    crmask = mask.ContiguousRandom(n_chars=4)
    rmask = mask.StochasticMask(probability=0.5)
    spm_tokenize = tokenize.SentencePieceTokenizer(args.spm_prefix)

    # Compute Batch Size
    max_tokens_per_device = 48000
    # max_tokens_per_device = 1000
    n_devices = torch.cuda.device_count()
    max_tokens = max_tokens_per_device * n_devices
    truncate_length = 20
    batch_size = int(max_tokens/truncate_length)

    checkpoint_path = "/home/jerin/mgan-attempts/"
    saver = Saver(checkpoint_path)

    train_path = os.path.join(args.path, 'train')
    dev_path = os.path.join(args.path, 'test')

    train_dataset = TensorIMDbDataset(
            train_path, spm_tokenize, 
            rmask, truncate_length
    )

    # Constructed vocabulary from train
    vocab = train_dataset.vocab
    Task = namedtuple('Task', 'source_dictionary target_dictionary')
    task = Task(source_dictionary=vocab, 
            target_dictionary=vocab)

    trainer = MGANTrainer(args, task, saver, visdom, vocab)
    def loader(dataset):
        _loader = DataLoader(dataset, batch_size=batch_size, 
                collate_fn=TensorIMDbDataset.collate, 
                shuffle=True, num_workers=8)
        return _loader

    #trainer.validate_dataset(loader(train_dataset))

    dev_dataset = TensorIMDbDataset(
            dev_path, spm_tokenize, 
            rmask, truncate_length,
            vocab 
    )

    Datasets = namedtuple('Dataset', 'train dev')
    datasets = Datasets(
            train=train_dataset,
            dev=dev_dataset
    )



    for epoch in tqdm(range(args.max_epochs), total=args.max_epochs, desc='epoch'):
        train_loader = loader(datasets.train)
        pbar = tqdm(train_loader, desc='training', leave=True)
        for i, samples in enumerate(pbar):
            trainer.run(epoch, samples)
            if i % args.validate_every == 0:
                validation_samples = 1000
                val_idxs = random.sample(range(len(datasets.dev)), validation_samples)
                subset = torch.utils.data.Subset(datasets.dev, val_idxs)
                trainer.validate_dataset(loader(subset))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--spm_prefix', required=True)
    parser.add_argument('--criterion', default='dummy')
    parser.add_argument('--max_epochs', type=int,  default=10)
    parser.add_argument('--validate_every', type=int,  default=5)
    parser.add_argument('--num_rollouts', type=int,  default=5)
    args = parser.parse_args()
    main(args)

