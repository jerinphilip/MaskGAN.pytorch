from torch import nn

class Tokenizer(nn.Module):
    pass

class SpaceTokenizer(Tokenizer):
    def forward(self, seq):
        return seq.split()
