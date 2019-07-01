import numpy as np
from . import glue
from . import observation

class NativeSamplingPool :
    def __init__(self, ptr, qreg_ordering, mask) :
        self.ptr = ptr
        self.qreg_ordering = qreg_ordering
        self.mask = mask
        
    def __del__(self) :
        self.delete()

    def delete(self) :
        if hasattr(self, 'ptr') :
            glue.sampling_pool_delete(self.ptr)
            del self.ptr

    def sample(self, n_samples, randnum = None) :
        obs = np.empty([n_samples], np.int64)
        if randnum is None :
            randnum = np.random.random_sample([n_samples])
        glue.sampling_pool_sample(self.ptr, obs, n_samples, randnum)
        return observation.ObservationList(self.qreg_ordering, obs, self.mask)
