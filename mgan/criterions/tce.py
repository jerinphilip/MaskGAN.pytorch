from torch import nn
import torch

class TBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_logits, truths, weight=None):
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')
        B, T, H = pred_logits.size()
        truths = truths.float()
        weight = weight.float()
        # _debug(pred_logits, truths, weight)
        loss = self.criterion(pred_logits, truths)
        return loss.sum()
        # missing = weight.sum().item()
        # assert (missing != 0)
        # return ((loss * weight).sum()/missing)

class TCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, truths, weight=None):
        logits = logits.contiguous()
        B, T, H = logits.size()
        logits = logits.view(T*B, H)
        target = truths.contiguous().view(-1)
        loss = self.criterion(logits, target)
        return loss

class WeightedMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='none')

    def forward(self, preds, truths, weights):
        mse_loss = self.criterion(preds, truths)
        # missing = weights.sum()
        return mse_loss
        return weights*mse_loss/missing


def _debug(pred_logits, truths, weight):
    B, T, H = pred_logits.size()
    # pred_logits = pred_logits.view(-1, H)
    # truths = truths.view(-1)
    for b in range(B):
        npreds = pred_logits[b, :, :].view(-1)
        ntruths = truths[b, :, :].view(-1)
        nweights = weight[b, :].view(-1)
        weighted = nn.BCEWithLogitsLoss(reduction='none')(npreds, ntruths) * nweights
        outstr = """
sizes: {} {} {}
predns:  {}
truths:  {}
weights: {}
final:   {}
        """.format(npreds.size(), ntruths.size(), nweights.size(),
                torch.sigmoid(npreds).tolist(),
                ntruths.tolist(),
                nweights.tolist(),
                weighted.tolist()
        )
        print(outstr, flush=True)
        break
