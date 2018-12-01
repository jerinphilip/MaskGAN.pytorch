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

class TensorIMDbDataset(Dataset):
    def __init__(self, path, tokenizer, mask_builder, truncate_length, rebuild=False):
        self.path = path
        self._dataset = IMDbEnhancedDataset(path, tokenizer, truncate_length)
        self.mask_builder = mask_builder
        self.tokenizer = tokenizer
        self.build_vocab(rebuild=rebuild)
        self.truncate_length = truncate_length

    def build_vocab(self, rebuild=False):
        vocab_path = os.path.join(self.path, 'vocab.pt')
        if os.path.exists(vocab_path) and not rebuild:
            self.vocab = Dictionary.load(vocab_path)
        else:
            self.rebuild_vocab()
    
    def rebuild_vocab(self):
        dataset = IMDbDataset(self.path)
        vocab_path = os.path.join(self.path, 'vocab.pt')
        self.vocab = Dictionary()
        self.vocab.add_symbol(self.mask_builder.mask_token)
        for i in tqdm(range(len(dataset)), desc='build-vocab'):
            contents = dataset[i]
            tokens = self.tokenizer(contents)
            for token in tokens:
                self.vocab.add_symbol(token)
        self.vocab.save(vocab_path)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        # contents = self._dataset[idx]
        tokens = self._dataset[idx]
        # tokens = self.tokenizer(contents)
        # print(self.truncate_length, len(tokens))
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
