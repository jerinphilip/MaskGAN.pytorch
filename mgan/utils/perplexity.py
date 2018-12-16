import torch
import pdb
import math

def greedy_sample(logits):
    batch_size, seq_len, _ = logits.size()
    max_values, max_indices = logits.max(dim=2)
    return max_indices
    # sampled = []
    # for t in range(seq_len):
    #     dt = logits[:, t, :]
    #     max_values, max_indices = dt.max(dim=1)
    #     sampled.append(max_indices)
    # return torch.stack(sampled, dim=1)

def ppl(sequences, log_probs):
    batch_size, seq_len = sequences.size()
    # sequences = sequences.view(-1).long()
    # seq_log_probs = log_probs[:, :, sequences]
    seq_log_probs = torch.zeros_like(sequences).float()
    for b in range(batch_size):
        for t in range(seq_len):
            idx = sequences[b, t].item()
            seq_log_probs[b, t] = log_probs[b, t, idx].item()
    return seq_log_probs.sum()

# def perplexity(masked, lengths, mask, unmasked, log_probs):
def perplexity(truths, sampled, log_probs):
    batch_size, seq_len, vocab_size = log_probs.size()
    # sampled = greedy_sample(log_probs)
    _ppl = {
        'ground-truth': ppl(truths, log_probs).mean(),
        'sampled': ppl(sampled, log_probs).mean(),
    }
    return _ppl


