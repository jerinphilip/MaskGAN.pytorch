import torchnet as tnt
import functools
from copy import deepcopy
from torchnet.logger import \
        VisdomPlotLogger, \
        VisdomLogger,   \
        VisdomTextLogger

import subprocess
from warnings import warn

def git_hash():
    command = 'git rev-parse --short HEAD'
    toks = command.split()
    output = subprocess.check_output(toks)
    _hash = output.strip().decode("ascii")
    return _hash

# Track a list of loggers, use meters.
# Track a list of meters.

class devnull:
    def __init__(self, *args, **kwargs): pass
    def log(self, *args, **kwargs): pass


class VisdomCentral:
    def __init__(self):
        self.devnull = devnull()
        self.defaults = {
            "server": "localhost",
            "port": 8097,
            "env": git_hash()
            # "env": "main"
        }

        self.loggers = {}
        self.init_loggers()

    def init_loggers(self):
        def plogger(title):
            return VisdomPlotLogger('line', opts={'title': title}, **self.defaults)

        keys = [
            'generator/loss',
            'discriminator/real',
            'discriminator',
            'discriminator/fake',
            'critic/loss',
            'critic/pretrain'
        ]

        self.loggers = dict([(k, plogger(k)) for k in keys])

    def log(self, key, *args):
        if key in self.loggers:
            self.loggers[key].log(*args)
        else:
            warn("Logger {logger} not registered.".format(logger=key))

visdom = VisdomCentral()
