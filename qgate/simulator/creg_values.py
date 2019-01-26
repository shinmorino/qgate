import qgate.model as model

# creg arrays and their values.
class CregValues :
    def __init__(self) :
        self.cregdict = dict()

    def add(self, cregs) :
        for creg in cregs :
            self.cregdict[creg.id] = 0
    
    def set(self, creg, value) :
        self.cregdict[creg.id] = value
        
    def get(self, cregs) :
        if isinstance(cregs, model.Creg) :
            return self.cregdict[cregs.id]
        return [self.cregdict[creg.id] for creg in cregs]

    def get_packed_value(self, creg_array) :
        ivalue = 0
        for idx, creg in enumerate(creg_array) :
            value = self.cregdict.get(creg.id, 0)
            if value == 1 :
                ivalue |= 1 << idx
        return ivalue
