import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from fairseq.data.dictionary import Dictionary
from tqdm import tqdm
from copy import deepcopy
from .imdb_dataset import IMDbDataset
import bisect

class IMDbEnhancedDataset(IMDbDataset):
    def __init__(self, path, tokenizer, truncate):
        self.dataset = IMDbDataset(path)
        self.tokenizer = tokenizer
        self._length = self.build_inverse_index(truncate)
        self.truncate = truncate

    def __len__(self):
        return self._length

    def build_inverse_index(self, n):
        N = len(self.dataset)
        pbar = tqdm(
          range(N), total=N,
          desc='building inv-idx', leave=False
        )

        self.cumulative = [0]
        previous = 0
        for idx in pbar:
            sample = self.dataset[idx]
            tokens = self.tokenizer(sample)
            count = max(0, len(tokens) - n)
            previous = previous + count
            self.cumulative.append(previous)
        return previous

    def __getitem__(self, idx):
        # Find the rightmost entry less than idx
        p_idx = bisect.bisect_right(self.cumulative, idx)
        j = idx - self.cumulative[p_idx-1]
        contents = self.dataset[p_idx-1]
        tokens = self.tokenizer(contents)
        segment = tokens[j:j+self.truncate]
        item = deepcopy(segment)
        return item
