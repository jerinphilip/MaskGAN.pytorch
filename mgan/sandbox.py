from mgan.data import IMDbDataset, TensorIMDbDataset
from argparse import ArgumentParser

def dataset_test(args):
    tokenize = lambda x: x.split()
    dataset = TensorIMDbDataset(args.path, tokenize)
    n = len(dataset)
    for i in range(n):
        print(dataset.__getitem__(i))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', required=True)
    args = parser.parse_args()
    dataset_test(args)

