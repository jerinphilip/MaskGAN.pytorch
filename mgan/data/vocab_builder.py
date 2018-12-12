import os
from fairseq.data.dictionary import Dictionary

class VocabBuilder:
    def __init__(self, dataset, tokenizer, save_path):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.vocab_path = os.path.join(save_path, 'vocab.pt')
        self._vocab = None

    def vocab(self):
        if self._vocab is None:
            self.build_vocab()
        return self._vocab

    def build_vocab(self):
        if os.path.exists(self.vocab_path):
            self._vocab = Dictionary.load(self.vocab_path)
        else:
            self.rebuild_vocab()
    
    def rebuild_vocab(self):
        self._vocab = Dictionary()
        self._vocab.add_symbol(self.mask_builder.mask_token)
        desc = 'build-vocab: {}'.format(self.save_path)
        pbar = tqdm(
                range(len(self.dataset)), 
                desc=desc, 
                leave=True
        )

        for i in pbar:
            contents = self.dataset[i]
            tokens = self.tokenizer(contents)
            for token in tokens:
                self._vocab.add_symbol(token)

        if self.save_path is not None:
            self._vocab.save(self.vocab_path)
