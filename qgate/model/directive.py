from .model import Operator

class ClauseBegin(Operator) :
    pass

class ClauseEnd(Operator) :
    pass

class NewQreg(Operator) :
    def __init__(self, qreg) :
        self.qreg = qreg

# FIXME: not directive.
class ReleaseQreg(Operator) :
    def __init__(self, qreg) :
        self.qreg = qreg

    def copy(self) :
        return ReleaseQreg(self.qreg)

class Join(Operator) :
    def __init__(self, qregs) :
        assert 1 < len(qregs)
        Operator.__init__(self)
        self.qreglist = list(qregs)

class Separate(Operator) :
    def __init__(self, qreg) :
        Operator.__init__(self)
        self.qreg = qreg

class NewReference(Operator) :
    def __init__(self, ref) :
        self.ref = ref
