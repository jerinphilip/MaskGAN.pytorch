import os
from fairseq.data.dictionary import Dictionary

class VocabBuilder:
    def __init__(self, dataset, tokenizer, save_path=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.save_path = save_path
        self._vocab = None
        self.build_vocab()

    @property
    def vocab(self):
        if self._vocab is None:
            self.build_vocab()
        return self._vocab

    @property
    def vocab_path(self):
        if self.save_path is None:
            return None
        return os.path.join(self.save_path, 'vocab.pt')

    def build_vocab(self):
        if self.vocab_path is not None:
            self._vocab = Dictionary.load(self.vocab_path)
        else:
            self.rebuild_vocab()
    
    def rebuild_vocab(self):
        self._vocab = Dictionary()
        self._vocab.add_symbol(self.mask_builder.mask_token)
        for i in tqdm(range(len(self.dataset)), desc='build-vocab'):
            contents = self.dataset[i]
            tokens = self.tokenizer(contents)
            for token in tokens:
                self._vocab.add_symbol(token)

        if self.save_path is not None:
            self._vocab.save(self.vocab_path)
