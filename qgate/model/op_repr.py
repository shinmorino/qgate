from . import model
from . import directive

def format_qreg(qreg) :
    if isinstance(qreg, model.Qreg) :
        return str(qreg.id)
    nums = [str(qreg.id) for qreg in qreg]
    qregstr = ','.join(nums)
    return qregstr

def format_ref(ref) :
    if isinstance(ref, model.Reference) :
        return str(ref.id)
    nums = [str(_ref.id) for _ref in ref]
    refstr = ','.join(nums)
    return refstr

def op_repr(self) :
    return self.__class__.__name__

def op_qreg_repr(self) :
    name = self.__class__.__name__
    qregstr = format_qreg(self.qreg)
    return '{}({})'.format(name, qregstr)

def op_qreglist_repr(self) :
    name = self.__class__.__name__
    qregstr = format_qreg(self.qreglist)
    return '{}({})'.format(name, qregstr)

def op_qregset_repr(self) :
    name = self.__class__.__name__
    qregstr = format_qreg(self.qregset)
    return '{}({})'.format(name, qregstr)

def op_ref_repr(self) :
    name = self.__class__.__name__
    refstr = format_ref(self.ref)
    return '{}({})'.format(name, refstr)

directive.ClauseBegin.__repr__ = op_repr
directive.ClauseEnd.__repr__ = op_repr
directive.NewQreg.__repr__ = op_qreg_repr
directive.ReleaseQreg.__repr__ = op_qreg_repr
directive.Join.__repr__ = op_qreglist_repr
directive.Separate.__repr__ = op_qreg_repr

directive.NewReference.__repr__ = op_ref_repr

model.Barrier.__repr__ = op_qregset_repr
model.Reset.__repr__ = op_qregset_repr


def gate_repr(self) :
    name = self.gate_type.__class__.__name__
    arglist = [str(arg) for arg in self.gate_type.args]
    params = ','.join(arglist)
    if len(params) != 0 :
        params = '(' + params + ')'
    adjoint = '.Adj' if self.adjoint else ''
    qregstr = format_qreg(self.qreg)
    ctrllist = ''
    if self.ctrllist is not None :
        ctrllist = 'cntr(' + format_qreg(self.ctrllist) + ').'
    return '{}{}{}{}({})'.format(ctrllist, name, params, adjoint, qregstr)

model.Gate.__repr__ = gate_repr

def composed_gate_repr(self) :
    name = self.gate_type.__class__.__name__
    arglist = [str(arg) for arg in self.gate_type.args]
    params = ','.join(arglist)
    if len(params) != 0 :
        params = '(' + params + ')'
    adjoint = '.Adj' if self.adjoint else ''
    qregstr = format_qreg(self.qreg)
    ctrllist = ''
    if self.ctrllist is not None :
        ctrllist = 'cntr(' + format_qreg(self.ctrllist) + ').'
    return '{}{}{}{}({})'.format(ctrllist, name, params, adjoint, qregstr)


#model.ComposedGate) :
#model.MultiQubitGate) :

def measure_repr(self) :
    qregstr = format_qreg(self.qreg)
    refstr = format_ref(self.outref)
    return 'Measure({}, {})'.format(refstr, qregstr)

model.Measure.__repr__ = measure_repr

def prob_repr(self) :
    qregstr = format_qreg(self.qreg)
    refstr = format_ref(self.outref)
    return 'Prob({}, {})'.format(refstr, qregstr)

model.Prob.__repr__ = prob_repr

def if_repr(self) :
    refstr = format_ref(self.refs)
    if callable(self.cond) :
        pred = repr(self.cond)
    else :
        pred = str(self.cond)
    
    return 'if(({}), {})'.format(refstr, pred)

model.IfClause.__repr__ = if_repr
