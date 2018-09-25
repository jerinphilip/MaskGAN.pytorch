from torch import nn

class TCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss
        pass

    def forward(self, predictions, truths):
        T, B, H = predictions.size()
        predictions = predictions.view(-1, H)
        truths = truths.view(-1)
        return self.criterion(predictions, truths)


