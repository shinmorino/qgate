import numpy as np
from . import glue

class NativeQubitStates :
    def __init__(self, ptr, processor) :
        self.ptr = ptr
        self.processor = processor
        
    def __del__(self) :
        self.delete()

    def delete(self) :
        if hasattr(self, 'ptr') :
            glue.qubit_states_delete(self.ptr)
            del self.ptr
        if hasattr(self, 'processor') :
            self.processor.delete()
            del self.processor

    def get_n_lanes(self) :
        return glue.qubit_states_get_n_lanes(self.ptr)
    
    def reset_lane_states(self) :
        self.lane_states = [-1] * self.get_n_lanes()
    
    def get_lane_state(self, lane) :
        return self.lane_states[lane]
    
    def set_lane_state(self, lane, value) :
        self.lane_states[lane] = value

    def calc_probability(self, lane) :
        return self.processor.calc_probability(self, lane)
