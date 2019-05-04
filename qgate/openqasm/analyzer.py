#from .parser import parse, node_props
import qgate.model as model

class Analyzer :
    def __init__(self, formatter) :
        self.qregdict = dict()
        self.cregdict = dict()
        self.formatter = formatter

    # FIXME: enable context checks.
    def add_qreg(self, name, num) :
        self.qregdict[name] = num

    def add_creg(self, name, num) :
        self.cregdict[name] = num

    def open_program(self) :
        self.formatter.prologue()

    def close_program(self) :
        self.formatter.epilogue()

    def decl(self, kw, id_, num) :
        if kw == 'qreg' :
            self.qregdict[id_] = int(num)
            self.formatter.qreg_decl(id_, num)
        else :
            assert kw == 'creg'
            self.cregdict[id_] = int(num)
            self.formatter.creg_decl(id_, num)

    def open_if(self, id_, nm) :
        self.formatter.open_if_clause(id_, nm)

    def close_if(self) :
        self.formatter.close_if_clause()

    def U_gate(self, explist, argument) :
        self.formatter.qop('U', explist, [argument])

    def CX_gate(self, arg0, arg1) :
        self.formatter.qop('CX', None, [arg0, arg1])

    def id_gate(self, gate_id, explist, anylist) :
        self.formatter.qop(gate_id, explist, anylist)

    def barrier(self, anylist) :
        self.formatter.barrier(anylist)

    def measure(self, arg0, arg1) :
        self.formatter.measure(arg0, arg1)

    def reset(self, arg) :
        self.formatter.reset(arg)
