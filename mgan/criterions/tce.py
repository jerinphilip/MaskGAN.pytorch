from torch import nn

class TCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        pass

    def forward(self, pred_logits, truths):
        T, B, H = pred_logits.size()
        pred_logits = pred_logits.view(-1, H)
        truths = truths.view(-1)
        return self.criterion(pred_logits, truths)


