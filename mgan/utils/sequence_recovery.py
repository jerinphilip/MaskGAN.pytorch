import torch

class SequenceGenerator:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, _input):
        return self.generate(_input)

    def generate(self, tensor):
        text = self.vocab.string(tensor)
        seqs = text.splitlines()
        return seqs

def pretty_print(vocab, masked, unmasked, generated):
    sequence_generator = SequenceGenerator(vocab)
    masked = sequence_generator(masked)
    unmasked = sequence_generator(unmasked)
    generated = sequence_generator(generated)
    lines = []
    for _masked, _unmasked, _generated in zip(masked, unmasked, generated):
        lines.append('> {}'.format(_masked))
        lines.append('< {}'.format(_generated))
        lines.append('= {}'.format(_unmasked))
        lines.append("")

    print('\n'.join(lines))




