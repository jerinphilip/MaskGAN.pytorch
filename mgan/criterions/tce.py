from torch import nn

class TCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_logits, truths, weight=None):
        self.criterion = nn.BCEWithLogitsLoss(weight=weight)
        B, T, H = pred_logits.size()
        # pred_logits = pred_logits.view(-1, H)
        # truths = truths.view(-1)
        pred_logits = pred_logits.view(B, -1).contiguous()
        truths = truths.view(B, -1)
        return self.criterion(pred_logits, truths)


