import torch 
device = torch.device("cuda")

def debug_generate(model, loader, vocab, visdom):
    from fairseq.sequence_generator import SequenceGenerator
    closure = lambda s: visdom.log("gen-output", "text-replace", s)
    seq_gen = SequenceGenerator([model], 
            vocab, beam_size=5)
    #pbar = tqdm_progress_bar(loader, epoch=epoch)
    for src, src_lens, _, tgt, tgt_lens, _ in loader:
        src = src.to(device)
        encoder_input = {"src_tokens": src, "src_lengths": src_lens}
        samples = seq_gen.generate(encoder_input, maxlen=20)
        all_lines = []
        for i, sample in enumerate(samples):
           src_str = vocab.string(src[i, :])
           tgt_str = vocab.string(tgt[i, :])
           pred_str = vocab.string(sample[0]['tokens'])
           lines = [
                       "> {}".format(src_str),
                       "< {}".format(pred_str),
                       "= {}".format(tgt_str),
                       ""
                   ]
           all_lines.extend(lines)
    model.train()

    txt_dump = '<br>'.join(all_lines[:100])
    #print('\n'.join(all_lines))
    closure(txt_dump)
