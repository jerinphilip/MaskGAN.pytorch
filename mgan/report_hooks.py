import torchnet as tnt
import functools
from copy import deepcopy

defaults = {
    "server": "10.2.16.179",
    "port": "8097"
}


# Track a list of loggers, use meters.
# Track a list of meters.

METER_REGISTRY = {}
VISDOM_REGISTRY = {}

def init_meters():
    METER_REGISTRY["time"] = tnt.meter.TimeMeter()
    METER_REGISTRY["train/epoch"] = tnt.meter.AverageValueMeter()
    METER_REGISTRY["valid/epoch"] = tnt.meter.AverageValueMeter()

def init_loggers():
    VISDOM_REGISTRY["train/epoch"] = VisdomPlotLogger('line', opts={'Title': 'Train Loss'}, **defaults)
    VISDOM_REGISTRY["valid/epoch"] = VisdomPlotLogger('line', opts={'Title': 'Valid Loss'}, **defaults)

init_meters()
init_loggers()


def add(_id, value):
    METER_REGISTRY[_id].add(value)

def flush(_id):
    mean, std = METER_REGISTRY[_id].value()
    VISDOM_REGISTRY[_id].log(mean)


