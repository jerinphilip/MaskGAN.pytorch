import torch
from torch.nn.parallel import DataParallel
from .distributed_model import MGANModel
from mgan.utils.sequence_recovery import pretty_print
from torch.nn.utils.clip_grad import clip_grad_norm_
import random


class ClippedAdam(torch.optim.Adam):
    def __init__(self, parameters, *args, **kwargs):
        super().__init__(parameters, *args, **kwargs)
        self.clip_value = 0
        self._parameters = parameters

    def set_clip(self, clip_value):
        self.clip = clip_value

    def step(self, *args, **kwargs):
        clip_grad_norm_(self._parameters, self.clip_value)
        super().step(*args, **kwargs)


class MGANTrainer:
    def __init__(self, args, task, saver, logger, vocab):
        device = torch.device("cuda")
        self.pretrain = False
        self._model = MGANModel.build_model(args, task, pretrain=self.pretrain)
        self.model = DataParallel(self._model)
        self.model = self.model.to(device)
        self.opt = ClippedAdam(self.model.parameters(), lr=1e-3)
        self.opt.set_clip(clip_value=5.0)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=0.5)
        self.saver = saver
        self.logger = logger
        self.step = 0
        self.vocab = vocab
        self.saver.load("mgan", self.model.module)
        self.critic_lag_max = 50
        self.critic_lag = self.critic_lag_max 

    def run(self, epoch, samples):
        # self._debug(samples)
        # return
        num_rollouts = 50
        # num_rollouts = 1
        self.lr_scheduler.step(epoch)
        # self.rollout_discriminator(num_rollouts=num_rollouts, samples=samples)
        self.rollout_generator(num_rollouts=num_rollouts, samples=samples)
        self.debug(samples)
        # self.rollout_critic(num_rollouts=num_rollouts, samples=samples)
        self.saver.checkpoint("mgan", self.model.module)
        self.step += 1

    def _debug(self, samples):
        masked, unmasked, lengths, mask = samples
        B, T = masked.size()
        for b in range(B):
            print("  masked:", masked[b, :].tolist())
            print("unmasked:", unmasked[b, :].tolist())
            print("    mask:", mask[b, :].tolist())
            print("")

    def rollout_discriminator(self, num_rollouts, samples):
        masked, unmasked, lengths, mask = samples
        d_real_loss, d_fake_loss = 0, 0,
        loss = 0
        self.opt.zero_grad()

        for rollout in range(num_rollouts):
            _d_real_loss, _ = self.model(masked, lengths, mask, unmasked, tag="d-step", real=True)

            with torch.no_grad():
                _gloss, generated, _closs, _ = self.model(masked, lengths, mask, unmasked, tag="g-step")

            # print("d-gen-vs-unmasked", unmasked.size(),  generated.size())
            _d_fake_loss, _  = self.model(masked, lengths, mask, generated, tag="d-step", real=False)


            _d_real_loss = _d_real_loss.mean()
            _d_fake_loss = _d_fake_loss.mean()

            loss = (_d_real_loss + _d_fake_loss )/2
            loss.backward()
            
            d_real_loss += _d_real_loss.item()
            d_fake_loss += _d_fake_loss.item()

        # loss.backward()
        self.opt.step()

        self.logger.log("discriminator/real", self.step, d_real_loss/num_rollouts)
        self.logger.log("discriminator/fake", self.step, d_fake_loss/num_rollouts)
        self.logger.log("discriminator",      self.step, (d_fake_loss+d_real_loss)/(2*num_rollouts))

    def rollout_critic(self, num_rollouts, samples):
        masked, unmasked, lengths, mask = samples
        closs = 0
        self.opt.zero_grad()

        for rollout in range(num_rollouts):
            if random.random() < 0.3:
                src_mask = torch.ones_like(src_mask)
            _gloss, samples, _closs, _ = self.model(masked, lengths, mask, unmasked, tag="g-step")
            #_closs = _closs.mean()
            loss += _closs.mean()
            closs += _closs.item()

        loss.backward()
        self.opt.step()
        self.logger.log("critic/pretrain", self.step, closs/num_rollouts)

    
    def rollout_generator(self, num_rollouts, samples):
        masked, unmasked, lengths, mask = samples

        gloss = 0
        closs = 0
        avg_reward = 0
        rgloss = 0
        rcloss = 0

        self.opt.zero_grad()
        for rollout in range(num_rollouts):
            _gloss, generated, _closs, _avg_reward = self.model(masked, lengths, mask, unmasked, tag="g-step")
            rgloss = _gloss.mean()
            gloss += _gloss.mean().item()

            if not self.pretrain:
                avg_reward += _avg_reward.mean().item()
                rcloss = _closs.mean()
                if self.critic_lag > 0:
                    self.critic_lag = self.critic_lag - 1
                    rcloss = rcloss*0 #zero out gradients
                else:
                    self.critic_lag = self.critic_lag_max

                rcloss.backward()
                closs += _closs.mean().item()

            rgloss.backward()

        self.opt.step()

        self.logger.log("generator/advantage", self.step, -1*gloss/num_rollouts)
        self.logger.log("generator/reward/token", self.step, avg_reward)
        self.logger.log("critic/loss", self.step, closs/num_rollouts)

    def debug(self, samples):
        masked, unmasked, lengths, mask = samples
        with torch.no_grad():
            _d_real_loss, _ = self.model(masked, lengths, mask, unmasked, tag="d-step", real=True)
            _gloss, generated, _closs, _ = self.model(masked, lengths, mask, unmasked, tag="g-step")
            _d_fake_loss, _  = self.model(masked, lengths, mask, generated, tag="d-step", real=False)
            pretty_print(self.vocab, masked, unmasked, generated, truncate=10)
