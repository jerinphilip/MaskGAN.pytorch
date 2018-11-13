from torch import nn
import sentencepiece as spm

class Tokenizer(nn.Module):
    pass

class SpaceTokenizer(Tokenizer):
    def forward(self, seq):
        return seq.split()


class SentencePieceTokenizer:
    def __init__(self, model_path):
        self.model_fpath = model_path
        self.sp = spm.SentencePieceProcessor() 
        self.sp.Load(self.model_fpath)

    def __call__(self, text):
        tokens = self.sp.EncodeAsPieces(text)
        return tokens

