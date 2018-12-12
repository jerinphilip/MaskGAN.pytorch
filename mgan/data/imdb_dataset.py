import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from fairseq.data.dictionary import Dictionary
from tqdm import tqdm
from copy import deepcopy

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
        with open(fpath) as fp:
            contents = fp.read()
            ignores = ['<br>', '<br/>', '<br />']
            for ignore in ignores:
                contents = contents.replace(ignore, '')
        return contents




