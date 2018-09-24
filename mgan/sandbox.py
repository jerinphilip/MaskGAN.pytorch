from mgan.data import IMDbDataset, TensorIMDbDataset
from argparse import ArgumentParser
from mgan.modules import Preprocess

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
    print(len(dataset.vocab))
    n = len(dataset)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', required=True)
    args = parser.parse_args()
    dataset_test(args)

