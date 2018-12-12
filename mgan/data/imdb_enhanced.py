import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from fairseq.data.dictionary import Dictionary
from tqdm import tqdm
from copy import deepcopy
from .imdb_dataset import IMDbDataset

class IMDbEnhancedDataset(IMDbDataset):
    def __init__(self, path, tokenizer, truncate):
        super().__init__(path)
        self.tokenizer = tokenizer
        self._length = self.build_inverse_index(truncate)
        self.truncate = truncate

    def __len__(self):
        return self._length

    def build_inverse_index(self, n):
        self.inverse_index = {}
        idy = 0

        pbar = tqdm(
          range(self.length), total=self.length,
          desc='building inv-idx', leave=False
        )

        for idx in pbar:
            sample = super().__getitem__(idx)
            tokens = self.tokenizer(sample)
            N = len(tokens)
            for j in range(N-n):
                self.inverse_index[idy] = (idx, j)
                idy += 1
        return (idy + 1)

    def __getitem__(self, idx):
        p_idx, j = self.inverse_index[idx]
        contents = super().__getitem__(p_idx)
        tokens = self.tokenizer(contents)
        return tokens[j:j+self.truncate]
