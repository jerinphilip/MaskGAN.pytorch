import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from fairseq.data.dictionary import Dictionary
from tqdm import tqdm
from copy import deepcopy
from .imdb_enhanced import IMDbEnhancedDataset
from .imdb_dataset import IMDbDataset
from .vocab_builder import VocabBuilder

class TensorIMDbDataset(Dataset):
    def __init__(self, path, tokenizer, mask_builder, truncate_length, vocab=None):
        self.path = path
        self._dataset = IMDbEnhancedDataset(path, tokenizer, truncate_length)
        self.mask_builder = mask_builder
        self.tokenizer = tokenizer
        self.truncate_length = truncate_length
        self.vocab = vocab
        self._construct_vocabulary()

    def _construct_vocabulary(self):
        if self.vocab is None:
            raw_dataset = IMDbDataset(self.path)
            builder = VocabBuilder(raw_dataset, self.tokenizer, self.path)
            self.vocab = builder.vocab

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        tokens = self._dataset[idx]
        sequence_length = min(self.truncate_length, len(tokens))
        mask_idxs = self.mask_builder(sequence_length)
        tokens = tokens[:sequence_length]

        def get_pair(tokens, mask_idxs, mask_id):
            idxs = [self.vocab.index(token) for token in tokens]

            def _pad(ls, desired_length, pad_index):
                padded_ls = deepcopy(ls)
                while len(padded_ls) <= desired_length:
                    padded_ls.append(pad_index)
                return padded_ls

            srcs = deepcopy(idxs)
            srcs.append(self.vocab.eos())

            tgts = deepcopy(idxs)
            tgts.insert(0, self.vocab.eos())

            srcs = _pad(srcs, self.truncate_length, self.vocab.pad())
            tgts = _pad(tgts, self.truncate_length, self.vocab.pad())

            mask = torch.zeros(len(tgts))
            for mask_idx in mask_idxs:
                offset = 1 # For eos
                mask[mask_idx + offset] = 1
                srcs[mask_idx] = mask_id

            return (srcs, tgts, len(srcs), mask)

        mask_id = self.vocab.index(self.mask_builder.mask_token)
        return get_pair(tokens, mask_idxs, mask_id)


    def get_collate_fn(self):
        return TensorIMDbDataset.collate

    @staticmethod
    def collate(samples):
        srcs, tgts, lengths, masks = list(zip(*samples))

        srcs = torch.LongTensor(srcs)
        tgts = torch.LongTensor(tgts)

        lengths = torch.LongTensor(lengths)
        lengths, sort_order = lengths.sort(descending=True)
        
        def _rearrange(tensor):
            return tensor.index_select(0, sort_order)

        srcs  = _rearrange(pad_sequence(srcs, batch_first=True))
        tgts  = _rearrange(pad_sequence(tgts, batch_first=True))
        masks = _rearrange(torch.stack(masks, dim=0))

        return (srcs, tgts, lengths, masks)
