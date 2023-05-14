"""
AIO -- All Trains in One
"""

from trains.baselines import *
from trains.missingTask import *

__all__ = ['ATIO']

class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            # single-task
            'mult': MULT,
            # missing-task
            'tfr_net': TFR_NET,
            'emt-dlfr': EMT_DLFR,
        }
    
    def getTrain(self, args):
        return self.TRAIN_MAP[args.modelName.lower()](args)
