
class Vocab:
    def __init__(self):
        self.w2i = {}
        self.i2w = {}
        self.counter = 0

        self.add('<s>')
        self.add('</s>')
        self.add('<unk>')

    def add(self, key):
        if key not in self.w2i:
            self.counter += 1
            self.w2i[key] = self.counter
            self.i2w[self.counter] = key

    def __getitem__(self, key):
        if key not in self.w2i:
            return self.w2i['<unk>']
        return self.w2i[key]


