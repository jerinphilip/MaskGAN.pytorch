from mgan.data import IMDbDataset, TensorIMDbDataset
from argparse import ArgumentParser
from mgan.modules import Preprocess

from torch.utils.data import DataLoader

def dataset_test(args):
    mask = {
        "type": "end",
        "kwargs": {"n_chars": 3}
    }

    tokenize = {
        "type": "space",
    }

    preprocess = Preprocess(mask, tokenize)
    dataset = TensorIMDbDataset(args.path, preprocess)
    loader = DataLoader(dataset, batch_size=12, collate_fn=TensorIMDbDataset.collate)
    for src, src_lens, tgt, tgt_lens in loader:
        print(src.size(), src_lens, tgt.size(), tgt_lens)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', required=True)
    args = parser.parse_args()
    dataset_test(args)

