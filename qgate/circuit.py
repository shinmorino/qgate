

class SubCircuit :
    def __init__(self) :
        pass

    
class Circuit :
    def __init__(self) :
        self.qreg_map = []
        self.creg_map = []
        self.sub_circuits = []
        self.ops = []

    def get_n_qregs(self) :
        return len(self.qreg_map)

    def get_n_cregs(self) :
        return len(self.creg_map)

    def map_qreg(self, qasm_qreg) :
        try :
            return self.qreg_map.index(qasm_qreg)
        except :
            self.qreg_map.append(qasm_qreg)
            return len(self.qreg_map) - 1
        
    def map_creg(self, creg) :
        try :
            return self.creg_map.index(creg)
        except :
            self.creg_map.append(creg)
            return len(self.creg_map) - 1

    def get_qreg_lane(self, qreg) :
        return self.qreg_map.index(qreg)
    
    def get_creg_lane(self, creg) :
        return self.creg_map.index(creg)

