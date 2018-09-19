from mgan.data import IMDbDataset
from argparse import ArgumentParser

def dataset_test(args):
    dataset = IMDbDataset(args.path)
    n = len(dataset)
    for i in range(n):
        print(dataset.__getitem__(i))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path', required=True)
    args = parser.parse_args()
    dataset_test(args)

