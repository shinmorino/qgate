from __future__ import print_function, unicode_literals
import sys

class Formatter :
    def __init__(self, file = sys.stdout, level = 0) :
        self.file = file
        self.level = level
        self.indent = ' ' * 4
        self.circuit_initialized = False
        self.circuit_list_open = False
        self.funcname = ''

    def set_funcname(self, funcname) :
        self.funcname = funcname;    

    def prologue(self) :
        self.write('import qgate')
        self.write('from qgate.script import *')
        self.write('import math')
        self.write('')
        
        if len(self.funcname) != 0 :
            self.write('def {}() :'.format(self.funcname))
            self.level += 1

    def epilogue(self) :
        if self.circuit_list_open :
            self.close_circuit_list()
            
        if len(self.funcname) != 0 :
            self.level -= 1

    def open_circuit_list(self) :
        self.circuit_list_open = True
        self.write('')
        if self.circuit_initialized :
            self.write('circuit += [')
        else :
            self.write('circuit = [')
            self.circuit_initialized = True
        self.level += 1

    def close_circuit_list(self) :
        self.circuit_list_open = False
        self.level -= 1
        self.write(']')
        self.write('')

    def open_if_clause(self, id_, nm) :
        self.write('if_({}, {}, ['.format(id_, nm))
        self.level += 1

    def close_if_clause(self) :
        self.level -= 1
        self.write('] ),')

    def qreg_decl(self, qreg_id, nregs) :
        if self.circuit_list_open :
            self.close_circuit_list()
        self.write('{} = new_qregs({})'.format(qreg_id, nregs))

    def creg_decl(self, creg_id, nregs) :
        if self.circuit_list_open :
            self.close_circuit_list()
        self.write('{} = new_references({})'.format(creg_id, nregs))

    def qop(self, gate_id, explist, arglist) :
        if not self.circuit_list_open :
            self.open_circuit_list()
        
        if gate_id == 'U' :
            self.write_gate(None, 'U3', explist, False, *arglist)
        elif gate_id == 'CX' :
            self.write_gate([arglist[0]], 'X', None, False, arglist[1])
        elif gate_id == 'swap' :
            self.write_gate([arglist[0]], 'Swap', None, False, arglist[1])
        elif gate_id in ['u3', 'u2', 'u1'] :
            self.write_gate(None, gate_id.upper(), explist, False, *arglist)
        elif gate_id in ['x', 'y', 'z', 'h', 's', 't']  :
            self.write_gate(None, gate_id.upper(), None, False, *arglist)
        elif gate_id in ['sdg', 'tdg'] :
            self.write_gate(None, gate_id[0].upper(), None, True, *arglist)
        elif gate_id == 'id' :
            self.write_gate(None, 'I', None, False, *arglist)
        elif gate_id in ['rx', 'ry'] :
            gate_id = 'R' + gate_id[1]
            self.write_gate(None, gate_id, explist, False, *arglist)
        elif gate_id == 'rz' :
            self.write_gate(None, 'U1', explist, False, *arglist)
        elif gate_id == 'cu1' :
            self.write_gate([arglist[0]], 'U1', explist, False, arglist[1])
        elif gate_id == 'cu3' :
            self.write_gate([arglist[0]], 'u3', explist, False, arglist[1])
        elif gate_id == 'ccx' :
            self.write_gate(arglist[0:2], 'X', None, False, arglist[2])
        elif gate_id in ['cx', 'cy', 'cz', 'ch'] :
            self.write_gate([arglist[0]], gate_id[1].upper(), None, False, arglist[1])
        else :
            assert False, 'unknown gate, {}.'.format(gate_id)

    def write_gate(self, ctrlargs, gate_id, explist, adjoint, *qreglist) :
        ctrl = ''
        if ctrlargs is not None :
            ctrl_repr = self.reg_repr_list(ctrlargs)
            ctrl = 'ctrl({}).'.format(','.join(ctrl_repr))
        params = ''
        if explist is not None :
            params = '(' + ','.join(explist) + ')'
        adj = ''
        if adjoint :
            adj = '.Adj'

        # each item is output for one element in qreglist
        for qreg in qreglist :
            if len(qreg) != 1 :
                # 1 indexed id
                qreg_repr = self.reg_repr(qreg)
                self.write('{}{}{}{}({}),'.format(ctrl, gate_id, params, adj, qreg_repr))
            else :
                qreg_id = self.reg_repr(qreg)
                var_name = '_' + qreg_id
                gatestr = '[{ctrl}{gate_id}{params}{adj}({var_name}) for {var_name} in {qreg_id}],'  \
                          .format(ctrl = ctrl, gate_id = gate_id, params = params, adj = adj, var_name = var_name, qreg_id = qreg_id)
                self.write(gatestr)

    def CX_gate(self, ctrl, target) :
        if not self.circuit_list_open :
            self.open_circuit_list()
        
        ctrl = self.reg_repr(ctrl)
        target = self.reg_repr(target)
        self.write('ctrl({}).X({}),'.format(ctrl, target))

    def barrier(self, qreglist) :
        if not self.circuit_list_open :
            self.open_circuit_list()
        
        for qreg in qreglist:
            if len(qreg) != 1 :
                # 1 indexed id
                qreg_repr = self.reg_repr(qreg)
                self.write('barrier({}),'.format(qreg_repr))
            else :
                qreg_id = self.reg_repr(qreg)
                var_qreg = '_' + qreg_id
                resetstr = '[barrier({var_qreg}) for {var_qreg} in {qreg_id}],'.format(var_qreg = var_qreg, qreg_id = qreg_id)
                self.write(resetstr)

    def reset(self, qreg) :
        if not self.circuit_list_open :
            self.open_circuit_list()
        
        qreg_repr = self.reg_repr(qreg)
        self.write('reset({}),'.format(qreg_repr))

    def measure(self, qreg, creg) :
        if not self.circuit_list_open :
            self.open_circuit_list()
        
        # FIXME: context check.  len(qreg) == len(creg)
        qreg_repr = self.reg_repr(qreg)
        creg_repr = self.reg_repr(creg)
        if len(qreg_repr) == 1 :
            qregvar = '_' + qreg_repr
            cregvar = '_' + creg_repr
            self.write('[measure({cvar}, {qvar}) for {cvar}, {qvar} in zip({creg_id}, {qreg_id})],'. \
                       format(cvar = cregvar, qvar = qregvar, creg_id = creg_repr, qreg_id = qreg_repr))
        else :
            self.write('measure({}, {}),'.format(creg_repr, qreg_repr))

    def reg_repr_list(self, regs) :
        formatted = []
        for reg in regs :
            reg_repr = self.reg_repr(reg)
            formatted.append(reg_repr)
        return formatted

    def reg_repr(self, reg) :
        if len(reg) == 1 :
            return reg[0]
        else :
            return '{}[{}]'.format(reg[0], reg[1])

    def write(self, str) :
        print(self.indent * self.level, end='', file = self.file)
        print(str, file = self.file)
