import os
import torch
from torch.utils.data import Dataset
from mgan.utils import Vocab

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
    def __init__(self, path, tokenize):
        super().__init__(path)
        self.tokenize = tokenize
        self.build_vocab()

    def build_vocab(self):
        self.vocab = Vocab()
        for i in range(self.length):
            contents = self.__getitem__(i)
            tokens = self.tokenize(contents)
            self.vocab.add(tokenize)


    def __getitem__(self, idx):
        contents = super.__getitem__(idx)
        tokens = self.tokenize(contents)
        idxs = []
        for token in tokens:
            idxs.append(self.vocab[token])
        return torch.LongTensor(idxs)

    @staticmethod
    def collate(self, samples):
        # TODO: Implement Collate
        pass
