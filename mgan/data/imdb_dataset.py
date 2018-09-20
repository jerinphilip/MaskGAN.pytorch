import os
import torch
from torch.utils.data import Dataset
from mgan.utils import Vocab
from tqdm import tqdm

class IMDbDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.precompute()

    def precompute(self):
        self.sample_files = []
        dirs = ['pos', 'neg', 'unsup']
        for _dir in dirs:
            path = os.path.join(self.path, _dir)
            for root, dirs, files in os.walk(path, topdown=False):
               for name in files:
                   fpath = os.path.join(root, name)
                   self.sample_files.append(fpath)

        self.length = len(self.sample_files)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        fpath = self.sample_files[idx]
        contents = open(fpath).read()
        ignores = ['<br>', '<br/>', '<br />']
        for ignore in ignores:
            contents = contents.replace(ignore, '')
        return contents

class TensorIMDbDataset(IMDbDataset):
    def __init__(self, path, preprocess, truncate=40):
        super().__init__(path)
        self.preprocess = preprocess
        self.truncate = truncate
        self.build_vocab()

    def build_vocab(self):
        self.vocab = Vocab()
        for i in tqdm(range(self.length), desc='build-vocab'):
            contents = super().__getitem__(i)
            tokens = self.preprocess(contents, mask=False)
            tokens = self._truncate(tokens)
            for token in tokens:
                self.vocab.add(token)
        self.vocab.add(self.preprocess.mask.mask_token)

    def _truncate(self, tokens):
        truncate = max(len(tokens), self.truncate)
        tokens = tokens[:truncate]
        return tokens


    def __getitem__(self, idx):
        contents = super().__getitem__(idx)
        tokens = self.preprocess(contents)
        tokens = self._truncate(tokens)
        idxs = []
        for token in tokens:
            idxs.append(self.vocab[token])
        return torch.LongTensor(idxs)

    @staticmethod
    def collate(self, samples):
        # TODO: Implement Collate
        pass

