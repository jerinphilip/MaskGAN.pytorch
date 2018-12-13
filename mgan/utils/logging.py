import torchnet as tnt
import functools
from copy import deepcopy
from torchnet.logger import \
        VisdomPlotLogger, \
        VisdomLogger,   \
        VisdomTextLogger
from visdom import Visdom

import subprocess
from warnings import warn
from collections import deque

def git_hash():
    command = 'git rev-parse --short HEAD'
    toks = command.split()
    output = subprocess.check_output(toks)
    _hash = output.strip().decode("ascii")
    return _hash

def launch_time():
    from time import gmtime, strftime
    from datetime import datetime

    return strftime("%Y-%m-%d %H:%M:%S", gmtime())

class devnull:
    def __init__(self, *args, **kwargs): 
        del args
        del kwargs
    def log(self, *args, **kwargs):
        del args
        del kwargs


class VisdomCentral:
    def __init__(self):
        self.devnull = devnull()
        self.defaults = {
            "server": "localhost",
            "port": 8097,
            "env": git_hash() + '-' + launch_time()
            # "env": "main"
        }

        # self.check_visdom_works()
        self.loggers = {}
        self.init_loggers()
        self.queue = deque()

    def check_visdom_works(self):
        viz = Visdom(server='http://'+self.defaults["server"], port=self.defaults["port"])
        try:
            assert (viz.check_connection())
        except:
            raise Exception("Error: Check Visdom Server Setup")

    def init_loggers(self):
        def plogger(title):
            return VisdomPlotLogger('line', opts={'title': title}, **self.defaults)

        keys = [
            'generator/advantage',
            'discriminator/real',
            'discriminator',
            'discriminator/fake',
            'critic/loss',
            'critic/pretrain',
            'generator/reward/token'
            'ppl/ground-truths'
            'ppl/sampled'
        ]

        self.loggers = dict([(k, plogger(k)) for k in keys])
        for key in ['train', 'dev']:
            tag = 'generated/{}'.format(key)
            self.loggers[tag] = VisdomTextLogger(update_type='REPLACE', **self.defaults)

    def log(self, key, *args):
        if key in self.loggers:
            closure = lambda: self.loggers[key].log(*args)
            self.queue.append(closure)
            try:
                self.flush_queue()
            except:
                warn("Visdom not properly setup!")
        else:
            warn("Logger {logger} not registered.".format(logger=key))

    def flush_queue(self):
        assert(len(self.queue) > 0)
        first = self.queue.popleft()
        try:
            first()
            while len(self.queue) < 0:
                closure = self.queue.popleft()
                closure()
        except:
            self.queue.appendleft(first)


visdom = VisdomCentral()
# visdom = devnull()

