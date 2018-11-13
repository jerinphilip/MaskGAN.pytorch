from torch import nn
import torch

# TODO(jerin): Need to fix this as per
# https://github.com/tensorflow/models/blob/master/research/maskgan/model_utils/model_losses.py

class REINFORCE(nn.Module):
    def __init__(self, gamma):
        self.gamma = gamma
        super().__init__()

    def forward(self, log_probs, logits, weight, baselines=None):
        EPS = 1e-7
        batch_size, seqlen, _ = logits.size()
        probs = torch.sigmoid(logits)
        rewards = torch.log(probs + EPS)

        rewards = rewards.squeeze(2)
        logits = logits.squeeze(2)
        # print(rewards.size(), logits.size(), weight.size())

        rewards = rewards * weight
        logits = logits * weight

        cumulative_rewards = []
        for t in range(seqlen):
            cum_value = rewards.new_zeros(batch_size)
            for s in range(t, seqlen):
                exp = float(s-t)
                k = (self.gamma ** exp)
                cum_value +=  weight[:, s]  * rewards[:, s] * log_probs[:, s]
            cumulative_rewards.append(cum_value)

        cumulative_rewards = torch.stack(cumulative_rewards, dim=1)
        missing = weight.sum()
        reward = cumulative_rewards.sum()/ missing
        return reward


