from torch import nn
import sentencepiece as spm

class Tokenizer(nn.Module):
    pass

class SpaceTokenizer(Tokenizer):
    def forward(self, seq):
        return seq.split()


class SentencePieceTokenizer:
    def __init__(self, model_prefix):
        self.prefix = model_prefix

        self.path = {}
        for key in ['model', 'vocab']:
            self.path[key] = '{}.{}'.format(self.prefix, key)

        self.sp = spm.SentencePieceProcessor() 
        self.sp.Load(self.path['model'])

        # Build vocabulary.
        self.build_vocabulary()

    def build_vocabulary(self):
        self.vocab = set()
        for line in open(self.path['vocab']):
            word, score = line.strip().split()
            self.vocab.add(word)


    def __call__(self, text):
        tokens = self.sp.EncodeAsPieces(text)

        to_utf = lambda x: x.decode("utf-8") 
        stokens = list(map(to_utf, tokens))

        wanted = lambda s: s in self.vocab
        stokens = list(filter(wanted, stokens))
        return stokens

