
import os
from mgan.preproc import tokenize
from argparse import ArgumentParser
from mgan.data import IMDbDataset
from tqdm import tqdm, trange

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--spm_prefix', required=True)
    parser.add_argument('--path', required=True)
    args = parser.parse_args()

    tokenizer = tokenize.SentencePieceTokenizer(model_prefix=args.spm_prefix)

    vocab_path = '{}.vocab'.format(args.spm_prefix)
    words = set()
    for line in open(vocab_path):
        word, score = line.strip().split()
        words.add(word)

    # print(words, len(words))

    train_path = os.path.join(args.path, 'train')
    dataset = IMDbDataset(train_path) 
    N = len(dataset)
    missings = set()

    pbar = trange(N)
    for i in pbar:
        contents = dataset[i]
        tokens = tokenizer(contents)
        missing = set(tokens).difference(words)
        missings = missings.union(missing)
        pbar.set_postfix(count=len(missings))

