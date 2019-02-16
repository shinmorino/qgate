import qgate.model as model

# values for references
class ValueStore :
    def __init__(self) :
        self.valuedict = dict()

    def add(self, refs) :
        for ref in refs :
            self.valuedict[ref.id] = None
    
    def set(self, ref, value) :
        self.valuedict[ref.id] = value
        
    def get(self, refs) :
        if isinstance(refs, model.Reference) :
            return self.valuedict[refs.id]
        return [self.valuedict[ref.id] for ref in refs]

    def get_packed_value(self, ref_array) :
        ivalue = 0
        for idx, ref in enumerate(ref_array) :
            value = self.valuedict.get(ref.id, 0)
            if value == 1 :
                ivalue |= 1 << idx
        return ivalue


class ValueStoreSetter :
    def __init__(self, value_store, outref) :
        self.value_store = value_store
        self.outref = outref

    def __call__(self, value) :
        self.value_store.set(self.outref, value)
