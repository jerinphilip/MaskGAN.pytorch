from torch import nn
import torch
from .reinforce import REINFORCE

def pretrain(model, opt):
    def _inner(src_tokens, src_lengths, prev_output_tokens):
        criterion = nn.CrossEntropyLoss()
        # Add ignore index somehow.
        opt.zero_grad()
        net_output = model.generator(src_tokens, src_lengths, prev_output_tokens)
        logits = net_output[0].float()
        logits = logits[:, :-1, :].contiguous()

        T, B, H = logits.size()
        logits = logits.view(T*B, -1)
        target = prev_output_tokens[:, 1:].contiguous().view(-1)
        loss = criterion(logits, target)
        print("Pretrain Loss", loss.item())
        loss.backward()
        opt.step()
    return _inner



def train(model, opt): 
    def _inner(src_tokens, src_lengths, prev_output_tokens):
        d_steps = 10
        g_steps = 1
        criterion = torch.nn.BCEWithLogitsLoss()

        # Train discriminator to find actual sentences
        def dis_train_real():
            print("d_steps:", d_steps)
            for step in range(d_steps):
                opt.zero_grad()
                logits, attn_scores = model.discriminator(
                        prev_output_tokens[:, 1:], src_lengths, 
                        prev_output_tokens)

                truths = torch.ones_like(logits)
                dloss = criterion(logits,  truths)
                print("Discriminator Real Loss:", dloss.item())
                dloss.backward()
                opt.step()

        dis_train_real()

        def gen_train_real():
            for step in range(g_steps):
                samples, log_probs, attns = model.generator(src_tokens, 
                        src_lengths, prev_output_tokens)
                logits, attn_scores = model.discriminator(samples, 
                        src_lengths, prev_output_tokens)
                # REINFORCE implementation
                # Build as a loss function?
                reinforce = REINFORCE(gamma=0.95)
                opt.zero_grad()
                E_R = reinforce(log_probs, logits)
                gloss = -1*E_R.mean()
                print("Generator Reward:", gloss.item())
                gloss.backward()
                opt.step()

                opt.zero_grad()
                logits, attn_scores = model.discriminator(samples, 
                        src_lengths, prev_output_tokens)

                truths = torch.zeros_like(logits)
                dloss = criterion(logits, truths)
                print("Discriminator Fake Loss:", dloss.item())
                dloss.backward()
                opt.step()

        gen_train_real()
    return _inner
