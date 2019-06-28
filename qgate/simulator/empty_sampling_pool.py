import numpy as np
from . import observation

class EmptySamplingPool :
    def __init__(self, qreg_ordering) :
        self.qreg_ordering = qreg_ordering

    def sample(self, n_samples) :
        values = np.ones([n_samples], dtype = np.int64)
        return observation.ObservationList(self.qreg_ordering, values, 0)

