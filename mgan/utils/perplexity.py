
class Perplexity:
    def __init__(self, model):
        self.model = model
        self.log_softmax = nn.LogSoftmax(dim=2)

    def __call__(self, src_tokens, src_lengths, prev_output_tokens):
        logits, attn_scores = self.model(src_tokens, src_lengths, prev_output_tokens)
        log_probs = log_softmax(logits)
        batch_size, seq_len, vocab_size = log_probs.size()

        def greedy_sample(logits):
            batch_size, seq_len, _ = logits.size()
            sampled = []
            for t in range(seq_len):
                dt = logits[:, t, :]
                max_values, max_indices = dt.max(dim=1)
                sampled.append(max_indices)
            sampled = torch.stack(sampled, dim=1)
            return sampled
        

        def ppl(sequences, log_probs):
            batch_size, seq_len = sequences.size()
            seq_log_probs = torch.zeros_like(sequences).float()
            for b in range(batch_size):
                for t in range(seq_len):
                    idx = sequences[b, t]
                    seq_log_probs[b, t] = log_probs[b, t]

            log_ppls = seq_log_probs.mean(dim=1)
            ppls = log_ppls.exp()
            return ppls

        _ppl = {
            'ground-truth': ppl(prev_output_tokens, log_probs),
            'sampled': ppl(sam,pled, log_probs)
        }
        return _ppl

