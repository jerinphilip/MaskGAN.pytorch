from mgan.data import IMDbDataset, TensorIMDbDataset
from argparse import ArgumentParser
from mgan.modules import Preprocess
from torch.utils.data import DataLoader
from mgan.models import MaskedMLE
from collections import namedtuple
from torch.nn import functional as F
from torch import optim
from fairseq.meters import AverageMeter
from fairseq.progress_bar import tqdm_progress_bar
from tqdm import tqdm
import torch
import os


class Args: 
    criterion = 'dummy'

def dataset_test(args):
    mask = {
        "type": "random",
        "kwargs": {"probability": 0.3}
    }

    tokenize = {
        "type": "space",
    }

    preprocess = Preprocess(mask, tokenize)
    dataset = TensorIMDbDataset(args.path, preprocess)
    loader = DataLoader(dataset, batch_size=25, collate_fn=TensorIMDbDataset.collate)
    Task = namedtuple('Task', 'source_dictionary target_dictionary')
    task = Task(source_dictionary=dataset.vocab, target_dictionary=dataset.vocab)

    meters = {}
    meters['epoch'] = AverageMeter()
    meters['loss'] = AverageMeter()

    device = 'cuda'

    def checkpoint(model, opt, checkpoint_path):
        _payload = {
            "model": model.state_dict(),
            "opt": model.state_dict()
        }

        with open(checkpoint_path, "wb+") as fp:
            torch.save(_payload, fp)

    def load(model, opt, checkpoint_path):
        _payload = torch.load(checkpoint_path)
        model.load_state_dict(_payload["model"])
        # opt.load_state_dict(_payload["opt"])


    args = Args()
    model = MaskedMLE.build_model(args, task)
    model = model.to(device)
    opt = optim.Adam(model.parameters())
    reduce = True
    max_epochs = 100

    checkpoint_path = "best_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        load(model, opt, checkpoint_path)

    for epoch in tqdm(range(max_epochs), total=max_epochs, desc='epoch'):
        pbar = tqdm_progress_bar(loader, epoch=epoch)
        meters["loss"].reset()
        for src, src_lens, tgt, tgt_lens in pbar:
            #print(src.size(), src_lens, tgt.size(), tgt_lens)
            opt.zero_grad()
            src, tgt = src.to(device), tgt.to(device)
            net_output = model(src, src_lens, tgt)
            lprobs = model.get_normalized_probs(net_output, log_probs=True)
            lprobs = lprobs.view(-1, lprobs.size(-1))
            target = tgt.view(-1)
            loss = F.nll_loss(lprobs, target, size_average=False, ignore_index=dataset.vocab.pad(),
                              reduce=reduce)
            loss.backward()
            meters['loss'].update(loss.item())
            pbar.log(meters)
            opt.step()

        avg_loss = meters["loss"].avg
        meters['epoch'].update(avg_loss)
        checkpoint(model, opt, checkpoint_path)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', required=True)
    args = parser.parse_args()
    dataset_test(args)

