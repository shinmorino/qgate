

class SimpleQubitStatesFactory :

    def __init__(self, pkg, dtype) :
        self.pkg = pkg

    def create(self, n_lanes, dtype, processor) :
        qstates = self.pkg.create_qubit_states(dtype)
        processor.initialize_qubit_states(qstates, n_lanes)
        return qstates

class MultiDeviceQubitStatesFactory :

    def __init__(self, pkg, n_lanes_per_chunk, device_ids) :
        self.pkg = pkg
        self.dtype = dtype
        self.n_lanes_per_chunk = n_lanes_per_chunk
        self.device_ids = device_ids

    def create(self, n_lanes, dtype, processor) :
            
        qstates = self.pkg.create_qubit_states(dtype)
        processor.initialize_qubit_states(qstates, n_lanes, self.n_lanes_per_chunk,
                                          device_ids);
        
        n_devices_consumed = self.n_lanes_per_chunk // n_lanes
        # rotate device_ids
        self.device_ids = self.device_ids[-n_devices_consumed:] + self.device_ids[:-n_devices_consumed]
        return qstates
