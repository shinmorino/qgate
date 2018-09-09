        
class Circuit :
    def __init__(self) :
        self.qreg_map = []
        self.creg_map = []
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

    def map_creg(self, qasm_creg) :
        try :
            return self.creg_map.index(qasm_creg)
        except :
            self.creg_map.append(qasm_creg)
            return len(self.creg_map) - 1
