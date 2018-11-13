import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from fairseq.data.dictionary import Dictionary
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
        with open(fpath) as fp:
            contents = fp.read()
            ignores = ['<br>', '<br/>', '<br />']
            for ignore in ignores:
                contents = contents.replace(ignore, '')
            return contents

class IMDbSingleDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.lines = open(self.path).read().splitlines()
        # self.precompute()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]

class TensorIMDbDataset(IMDbSingleDataset):
    def __init__(self, path, preprocess, truncate=40, rebuild=False):
        super().__init__(path)
        self.preprocess = preprocess
        self.truncate = truncate
        self.build_vocab(rebuild=rebuild)

    def build_vocab(self, rebuild=False):
        ## vocab_path = os.path.join(self.path + '.vocab.pt')
        vocab_path = self.path + '.vocab.pt'
        if os.path.exists(vocab_path) and not rebuild:
            self.vocab = Dictionary.load(vocab_path)
        else:
            self.rebuild_vocab()
    
    def rebuild_vocab(self):
        vocab_path = self.path + '.vocab.pt'
        self.vocab = Dictionary()
        self.vocab.add_symbol(self.preprocess.mask.mask_token)
        for i in tqdm(range(len(self)), desc='build-vocab'):
            contents = super().__getitem__(i)
            tokens, mask = self.preprocess(contents, mask=False)
            tokens, token_count = self._truncate(tokens)
            for token in tokens:
                self.vocab.add_symbol(token)

        self.vocab.save(vocab_path)

    def _truncate(self, tokens):
        truncate = min(len(tokens), self.truncate)
        tokens = tokens[:truncate]
        token_count = len(tokens)
        while len(tokens) < self.truncate:
            tokens.append(self.vocab.pad())
        return (tokens, token_count)


    def __getitem__(self, idx):
        contents = super().__getitem__(idx)
        tgt, tgt_length, tgt_mask = self.Tensor_idxs(contents, masked=False, move_eos_to_beginning=True)
        src, src_length, src_mask  = self.Tensor_idxs(contents, masked=True)
        #assert(tgt_length == src_length)
        return (src, src_length, src_mask, tgt, tgt_length, tgt_mask)
    
    def Tensor_idxs(self, contents, masked=True, move_eos_to_beginning=False):
        tokens, tmask = self.preprocess(contents, mask=masked)
        tokens, token_count = self._truncate(tokens)
        
        idxs = []
        if move_eos_to_beginning:
            mask = torch.zeros(len(tokens)+2+1)
            idxs.append(self.vocab.eos())
            mask[1:token_count+1] = tmask[:token_count]
            token_count += 1
        else:
            mask = torch.zeros(len(tokens)+1+1)
            mask[:token_count] = tmask[:token_count]

        for token in tokens:
            idxs.append(self.vocab.index(token))

        idxs.append(self.vocab.eos())
        token_count += 1

        return (torch.LongTensor(idxs), token_count, mask)

    @staticmethod
    def collate(samples):
        # TODO: Implement Collate
        srcs, src_lengths, src_masks, \
                tgts, tgt_lengths, tgt_masks = list(zip(*samples))

        src_lengths = torch.LongTensor(src_lengths)
        tgt_lengths = torch.LongTensor(tgt_lengths)

        src_lengths, sort_order = src_lengths.sort(descending=True)
        tgt_lengths = tgt_lengths.index_select(0, sort_order)

        srcs = pad_sequence(srcs)
        tgts = pad_sequence(tgts)

        srcs = srcs.index_select(1, sort_order)
        tgts = tgts.index_select(1, sort_order)

        # TODO(jerin): Fix this.
        src_masks = torch.stack(src_masks).permute(1, 0).contiguous()
        src_masks = src_masks.index_select(1, sort_order)
        src_masks = src_masks.permute(1, 0).contiguous()

        tgt_masks = torch.stack(tgt_masks).permute(1, 0).contiguous()
        tgt_masks = tgt_masks.index_select(1, sort_order)
        tgt_masks = tgt_masks.permute(1, 0).contiguous()

        # src_masks = None
        # tgt_masks = None

        batch_first = True

        if batch_first:
            srcs = srcs.permute(1, 0).contiguous()
            tgts = tgts.permute(1, 0).contiguous()

        # print(srcs.size(), src_lengths, tgts.size(), tgt_lengths)
        return (srcs, src_lengths, src_masks, tgts, tgt_lengths, tgt_masks)


