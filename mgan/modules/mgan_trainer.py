import torch
from torch.nn.parallel import DataParallel
from .distributed_model import MGANModel


class MGANTrainer:
    def __init__(self, args, task, saver, logger):
        device = torch.device("cuda")
        self.pretrain = False
        self._model = MGANModel.build_model(args, task, pretrain=self.pretrain)
        self.model = DataParallel(self._model)
        self.model = self.model.to(device)
        self.opt = torch.optim.Adam(self.model.parameters())
        self.saver = saver
        self.logger = logger
        self.step = 0
        self.saver.load("mgan", self.model.module)


    def run(self, epoch, samples):
        g_steps, d_steps = 20, 20
        # self.run_gsteps(g_steps, samples)
        # self.run_dsteps(d_steps, samples)
        self.enhance_critic(samples)
        self.saver.checkpoint("mgan", self.model.module)
        self.step += 1

    def run_dsteps(self, d_steps, samples):
        src_tokens, src_lengths, src_mask, \
            tgt_tokens, tgt_lengths, tgt_mask = samples

        prev_output_tokens = tgt_tokens
        d_real_loss, d_fake_loss = 0, 0,
        for step in range(d_steps):
            self.opt.zero_grad()
            _d_real_loss, _ = self.model(prev_output_tokens[:, 1:], 
                    src_lengths, tgt_mask, prev_output_tokens, 
                    tag="d-step", real=True)
            _d_real_loss = _d_real_loss.mean()

            with torch.no_grad():
                _gloss, samples, _closs = self.model(src_tokens, src_lengths, src_mask,
                                prev_output_tokens, tag="g-step")

            _d_fake_loss, _  = self.model(samples, src_lengths, tgt_mask,
                             prev_output_tokens, 
                             tag="d-step", real=False)
            _d_fake_loss = _d_fake_loss.mean()

            loss = (_d_real_loss + _d_fake_loss )/2
            loss.backward()
            
            d_real_loss += _d_real_loss.item()
            d_fake_loss += _d_fake_loss.item()
            self.opt.step()

        self.logger.log("discriminator/real", self.step, d_real_loss/d_steps)
        self.logger.log("discriminator/fake", self.step, d_real_loss/d_steps)
        self.logger.log("discriminator",      self.step, (d_fake_loss+d_real_loss)/(2*d_steps))

    def enhance_critic(self, samples):
        src_tokens, src_lengths, src_mask, \
            tgt_tokens, tgt_lengths, tgt_mask = samples
        max_steps = 4
        closs = 0
        for steps in range(max_steps):
            _gloss, samples, _closs = self.model(src_tokens, src_lengths, torch.ones_like(src_mask),
                    tgt_tokens, tag="g-step")
            _closs = _closs.mean()
            _closs.backward()
            closs += _closs.item()
        self.logger.log("critic/pretrain", self.step, closs/steps)

    
    def run_gsteps(self, g_steps, samples):
        src_tokens, src_lengths, src_mask, \
            tgt_tokens, tgt_lengths, tgt_mask = samples

        prev_output_tokens = tgt_tokens
        gloss = 0
        closs = 0

        for step in range(g_steps):
            self.opt.zero_grad()
            _gloss, samples, _closs = self.model(src_tokens, src_lengths, src_mask,
                    prev_output_tokens, tag="g-step")

            _gloss = _gloss.mean()
            _gloss.backward()
            gloss += _gloss.item()

            if not self.pretrain:
                _closs = _closs.mean()
                _closs.backward()
                closs += _closs.item()

            self.opt.step()

        self.logger.log("generator/loss", self.step, gloss/g_steps)
        self.logger.log("critic/loss", self.step, closs/g_steps)
