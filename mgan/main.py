# General Imports
from argparse import ArgumentParser
from tqdm import tqdm
from collections import namedtuple
import gc

# Torch imports
import torch
from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader


# FairSeq imports
from fairseq.meters import AverageMeter
from fairseq.progress_bar import tqdm_progress_bar

from mgan.preproc import Preprocess, mask, tokenize
from mgan.data import IMDbDataset, TensorIMDbDataset
from mgan.modules import MGANTrainer
from mgan.utils import Saver
from mgan.utils.debug_generate import debug_generate
from mgan.utils.logging import visdom

class Args: 
    criterion = 'dummy'

def dataset_test(args):
    crmask = mask.ContiguousRandom(n_chars=4)
    rmask = mask.StochasticMask(probability=0.5)
    spm_tokenize = tokenize.SentencePieceTokenizer(model_path=args.spm_path)

    # Compute Batch Size
    max_tokens_per_device = 48000
    n_devices = torch.cuda.device_count()
    max_tokens = max_tokens_per_device * n_devices
    truncate_length = 20
    batch_size = int(max_tokens/truncate_length)

    dataset = TensorIMDbDataset(args.path, spm_tokenize, rmask, truncate_length, rebuild=False)
    loader = DataLoader(dataset, batch_size=batch_size, 
            collate_fn=dataset.get_collate_fn(), 
            shuffle=True, num_workers=16)

    Task = namedtuple('Task', 'source_dictionary target_dictionary')
    task = Task(source_dictionary=dataset.vocab, 
            target_dictionary=dataset.vocab)

    device = torch.device('cuda')
    args = Args()
    max_epochs = 1000

    checkpoint_path = "/home/jerin/mgan-attempts/"
    saver = Saver(checkpoint_path)
    trainer = MGANTrainer(args, task, saver, visdom, dataset.vocab)
    from mgan.utils.leaks import leak_check, LeakCheck

    # loader = [next(iter(loader))]

    for epoch in tqdm(range(max_epochs), total=max_epochs, desc='epoch'):
        pbar = tqdm_progress_bar(loader, epoch=epoch)
        for samples in pbar:
            with LeakCheck(flag=False):
                trainer.run(epoch, samples)
                gc.collect()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', required=True)
    parser.add_argument('--spm_path', required=True)
    args = parser.parse_args()
    dataset_test(args)

