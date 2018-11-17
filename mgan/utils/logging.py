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

        self.check_visdom_works()
        self.loggers = {}
        self.init_loggers()

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
        ]

        self.loggers = dict([(k, plogger(k)) for k in keys])

    def log(self, key, *args):
        if key in self.loggers:
            self.loggers[key].log(*args)
        else:
            warn("Logger {logger} not registered.".format(logger=key))

visdom = VisdomCentral()
