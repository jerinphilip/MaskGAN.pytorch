import torch
from collections import namedtuple, defaultdict
from torch.nn.parallel import DataParallel
from .distributed_model import MGANModel
from mgan.utils.sequence_recovery import pretty_print
from mgan.optim import ClippedAdam
import random
from tqdm import tqdm
from fairseq.meters import AverageMeter

class MGANTrainer:
    def __init__(self, args, task, saver, logger, vocab):
        device = torch.device("cuda")
        self.pretrain = False
        self.saver = saver
        self.logger = logger
        self._model = MGANModel.build_model(args, task, pretrain=self.pretrain)
        self.model = DataParallel(self._model)
        self.model = self.model.to(device)
        self.opt = ClippedAdam(self.model.parameters(), lr=1e-3)
        self.opt.set_clip(clip_value=5.0)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.opt, gamma=0.5)
        self.saver.load("mgan", self.model.module)
        self.step = 0
        self.vocab = vocab
        self.critic_lag_max = 50
        self.critic_lag = self.critic_lag_max 


    def run(self, epoch, samples):
        self.model.train()
        num_rollouts = 1 if self.pretrain else 50
        self.lr_scheduler.step(epoch)
        self.rollout_discriminator(num_rollouts, samples)
        self.rollout_generator(num_rollouts, samples)
        self.rollout_critic(num_rollouts, samples)
        self.saver.checkpoint("mgan", self.model.module)
        self.step += 1

    def rollout_discriminator(self, num_rollouts, samples):
        masked, unmasked, lengths, mask = samples
        real, fake = AverageMeter(), AverageMeter()
        batch_size, seq_len = samples[0].size()

        self.opt.zero_grad()
        pbar = tqdm(range(num_rollouts), 
                desc='discriminator-rollout', total=num_rollouts)

        for rollout in pbar:
            real_loss = self.model(
                    masked, lengths, mask, unmasked, 
                    tag="d-step", real=True
            )

            real_loss = real_loss.sum()/batch_size

            with torch.no_grad():
                net_output = self.model(
                        masked, lengths, mask, 
                        unmasked, tag="g-step"
                )
                generated = net_output[1]

            fake_loss = self.model(
                    masked, lengths, mask, generated, 
                    tag="d-step", real=False
            )

            fake_loss = fake_loss.sum()/batch_size

            loss = (real_loss + fake_loss)/2
            loss.backward()

            real.update(real_loss.item())
            fake.update(fake_loss.item())

        self.opt.step()
        self.logger.log("discriminator/real", self.step, real.avg)
        self.logger.log("discriminator/fake", self.step, fake.avg)
        self.logger.log("discriminator",      self.step, real.avg + fake.avg)

    def rollout_critic(self, num_rollouts, samples):
        masked, unmasked, lengths, mask = samples
        batch_size, seq_len = samples[0].size()
        meter = AverageMeter()
        self.opt.zero_grad()
        pbar = tqdm(range(num_rollouts), 
                desc='critic-rollout', total=num_rollouts)
        for rollout in pbar:
            loss = self.model(masked, lengths, mask, unmasked, tag="c-step")
            loss = loss.sum() / batch_size
            loss.backward()
            meter.update(loss.item())

        self.opt.step()
        self.logger.log("critic/loss", self.step, meter.avg)

    
    def rollout_generator(self, num_rollouts, samples):
        masked, unmasked, lengths, mask = samples
        batch_size, seq_len = samples[0].size()
        meter = AverageMeter()
        ppl_meter = defaultdict(lambda: AverageMeter())
        self.opt.zero_grad()
        pbar = tqdm(range(num_rollouts), 
                desc='generator-rollout', total=num_rollouts)

        for rollout in pbar:
            loss, generated, ppl = self.model(masked, lengths, mask, unmasked, tag="g-step")
            loss = loss.sum() / batch_size
            loss.backward()
            meter.update(-1*loss.item())
            for key in ppl:
                ppl[key] = ppl[key].sum() / batch_size
                ppl_meter[key].update(ppl[key].item())
        self.opt.step()
        self.logger.log("generator/advantage", self.step, meter.avg)
        for key in ppl_meter:
            self.logger.log("ppl/{}".format(key), ppl_meter[key].avg)

        self.debug('train', samples, generated)

    def debug(self, key, samples, generated):
        masked, unmasked, lengths, mask = samples
        tag = 'generated/{}'.format(key)
        logger = lambda s: self.logger.log(tag, s)
        pretty_print(logger, self.vocab, masked, unmasked, generated, truncate=10)

    def validate_dataset(self, loader):
        self.model.eval()
        _meters = 'generator dfake dreal critic ppl_sampled ppl_truths'
        _n_meters = len(_meters.split())
        Meters = namedtuple('Meters', _meters)
        meters_list =  [AverageMeter() for i in range(_n_meters)]
        meters = Meters(*meters_list)
        for sample_batch in loader:
            self._validate(meters, sample_batch)
            for key, value in meters._asdict().items():
                print(key, value.avg)

    @property
    def umodel(self):
        if isinstance(self.model, DataParallel):
            return self.model.module
        return self.model

    def aggregate(self, batch_size):
        return lambda tensor: tensor.sum()/batch_size

    def _validate(self, meters, samples):
        with torch.no_grad():
            masked, unmasked, lengths, mask = samples
            batch_size, seq_len = samples[0].size()

            agg = self.aggregate(batch_size)

            real_loss = self.model(
                    masked, lengths, mask, unmasked, 
                    tag="d-step", real=True
            )

            real_loss = agg(real_loss)

            generator_loss, generated, ppl = self.model(
                    masked, lengths, mask, 
                    unmasked, tag="g-step"
            )

            generator_loss = agg(generator_loss)

            fake_loss = self.model(
                    masked, lengths, mask, generated, 
                    tag="d-step", real=False
            )

            fake_loss = agg(fake_loss)

            loss = (real_loss + fake_loss)/2

            critic_loss = self.model(masked, lengths, mask, unmasked, tag="c-step")
            critic_loss = agg(fake_loss)

            meters.dreal.update(real_loss.item())
            meters.dfake.update(fake_loss.item())
            meters.generator.update(generator_loss.item())
            meters.critic.update(critic_loss.item())

            for key in ppl:
                ppl[key] = agg(ppl[key])

            meters.ppl_sampled.update(ppl['sampled'].item())
            meters.ppl_truths.update(ppl['ground-truth'].item())
            self.debug('dev', samples, generated)
