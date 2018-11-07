import torchnet as tnt
import functools
from copy import deepcopy
from torchnet.logger import \
        VisdomPlotLogger, \
        VisdomLogger,   \
        VisdomTextLogger


# Track a list of loggers, use meters.
# Track a list of meters.

class devnull:
    def __init__(self, *args, **kwargs): pass
    def log(self, *args, **kwargs): pass


class VisdomCentral:
    def __init__(self, defaults):
        self.loggers = {}
        self.devnull = devnull()
        self.counter = {}

    def log(self, tag, _type, value):
        # print(tag, _type, value)
        logger = self.get_logger(tag, _type)
        if _type == 'line':
            self.counter[tag] += 1
            logger.log(self.counter[tag], value)
        elif _type == 'text-append':
            logger.log(value)

    def get_logger(self, tag, _type):
        if tag not in self.loggers:
            self.loggers[tag] = self._create_logger(_type)
            self.counter[tag] = 0
        return self.loggers[tag]
    
    def _create_logger(self, _type):
        _dict = {
                "line": VisdomPlotLogger('line', 
                                 opts={'title': 'Train Loss'}, 
                                 **defaults),
                "text-append": VisdomTextLogger(update_type='APPEND')
        }

        return _dict.get(_type, self.devnull)


defaults = {
    "server": "localhost",
    "port": "8097",
    "env": "main",
}

visdom = VisdomCentral(defaults)
