import qgate.model as model

# values for references
class ValueStore :
    def __init__(self) :
        self.valuedict = dict()

    def reset(self) :
        self.valuedict = dict()
        
    def sync_refs(self, refs) :
        for ref in refs :
            # check already added.
            if self.valuedict.get(ref.id, None) is None :
                # initial value is None.
                self.valuedict[ref.id] = None
    
    def set(self, ref, value) :
        self.valuedict[ref.id] = value
        
    def get(self, refs) :
        if isinstance(refs, model.Reference) :
            return self.valuedict.get(refs.id, None)
        return [self.valuedict.get(ref.id, None) for ref in refs]

    def get_packed_value(self, ref_array) :
        ivalue = 0
        for idx, ref in enumerate(ref_array) :
            # None is treated as 0 according to OpenQASM.
            value = self.valuedict.get(ref.id, 0)
            if value == 1 :
                ivalue |= 1 << idx
        return ivalue

    def get_mask(self, ref_array) :
        mask = 0
        for idx, ref in enumerate(ref_array) :
            if not ref.id in self.valuedict :
                mask |= 1 << idx
        return mask


class ValueStoreSetter :
    def __init__(self, value_store, outref) :
        self.value_store = value_store
        self.outref = outref

    def __call__(self, value) :
        self.value_store.set(self.outref, value)
