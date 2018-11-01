from torch import nn
import torch

class REINFORCE(nn.Module):
    def __init__(self, gamma):
        self.gamma = gamma
        super().__init__()

    def forward(self, log_probs, logits):
        batch_size, seqlen, _ = logits.size()
        probs = torch.sigmoid(logits)
        r = torch.log(probs)
        R = [0 for i in range(seqlen+1)]
        gamma_t = self.gamma
        for t in reversed(range(seqlen)):
            R[t] = gamma_t * r[:, t] + R[t+1]
            gamma_t = gamma_t*self.gamma

        E_R = 0
        for t in range(seqlen-11):
            E_R += R[t]*log_probs[t]

        return E_R

