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
from fairseq.sequence_generator import SequenceGenerator
from tqdm import tqdm
import torch
import os
from torch.nn import DataParallel
from torch import nn
from mgan.modules.generator import Generator, LossGenerator


from mgan.models.mgan import MaskGAN


class Args: 
    criterion = 'dummy'

def dataset_test(args):
    mask = {
        "type": "random",
        "kwargs": {"probability": 0.4}
    }

    tokenize = {
        "type": "space",
    }

    preprocess = Preprocess(mask, tokenize)
    dataset = TensorIMDbDataset(args.path, preprocess, truncate=20)
    loader = DataLoader(dataset, batch_size=220, collate_fn=TensorIMDbDataset.collate, shuffle=True, num_workers=16)
    Task = namedtuple('Task', 'source_dictionary target_dictionary')
    task = Task(source_dictionary=dataset.vocab, target_dictionary=dataset.vocab)

    meters = {}
    meters['epoch'] = AverageMeter()
    meters['loss'] = AverageMeter()

    device = torch.device('cuda')

    def checkpoint(model, opt, checkpoint_path):
        _payload = {
            "model": model.module.state_dict(),
            "opt": opt.state_dict()
        }

        with open(checkpoint_path, "wb+") as fp:
            torch.save(_payload, fp)

    def load(model, opt, checkpoint_path):
        _payload = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        #_payload = torch.load(checkpoint_path)
        model.module.load_state_dict(_payload["model"])
        opt.load_state_dict(_payload["opt"])


    args = Args()
    # model = MaskedMLE.build_model(args, task)
    model = MaskGAN.build_model(args, task)
    reduce = True
    max_epochs = 1


    criterion = nn.NLLLoss(ignore_index=dataset.vocab.pad())
    model = LossGenerator(model, criterion)
    checkpoint_path = "/scratch/jerin/best_checkpoint.pt"
    model = DataParallel(model, output_device=2)
    opt = optim.Adam(model.parameters())
    model = model.to(device)
    # if os.path.exists(checkpoint_path):
    #    load(model, opt, checkpoint_path)



    for epoch in tqdm(range(max_epochs), total=max_epochs, desc='epoch'):
        pbar = tqdm_progress_bar(loader, epoch=epoch)
        meters["loss"].reset()
        count = 0
        for src, src_lens, tgt, tgt_lens in pbar:
            count += 1
            opt.zero_grad()
            src, tgt = src.to(device), tgt.to(device)
            loss = model(src, src_lens, tgt)
            loss.sum().backward()
            meters['loss'].update(loss.mean().item())
            pbar.log(meters)
            opt.step()


        avg_loss = meters["loss"].avg
        meters['epoch'].update(avg_loss)
        checkpoint(model, opt, checkpoint_path)

    seq_gen = SequenceGenerator([model.module.model], dataset.vocab, beam_size=5)
    for src, src_lens, tgt, tgt_lens in loader:
        src = src.to(device)
        encoder_input = {"src_tokens": src, "src_lengths": src_lens}
        samples = seq_gen.generate(encoder_input, maxlen=20)
        for i, sample in enumerate(samples):
           # print(sample[0].keys())
           src_str = dataset.vocab.string(src[i, :])
           tgt_str = dataset.vocab.string(tgt[i, :])
           pred_str = dataset.vocab.string(sample[0]['tokens'])
           print(">", src_str)
           print("<", pred_str)
           print("=", tgt_str)
           print("")
        break



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', required=True)
    args = parser.parse_args()
    dataset_test(args)

