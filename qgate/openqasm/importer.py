from .analyzer import Analyzer
from .formatter import Formatter
from . import yacc
import io
import imp
import sys
import traceback
from . import yacc

def translate(qasm) :
    strfile = io.StringIO()
    formatter = Formatter(file=strfile)
    # formatter.set_funcname(factory_name)
    analyzer = Analyzer(formatter)
    yacc.parse(qasm, analyzer)
    code = strfile.getvalue()
    strfile.close()
    return code

def translate_file(filename) :
    with open(filename, 'r') as file :
        qasm = file.read()
    return translate(qasm)

def load_circuit(qasm) :
    code = translate(qasm)
    module = imp.new_module('imported')
    errmsg = None
    try :
        exec(code, module.__dict__)
    except Exception as e:
        detail = e.args[0]
        cl, exc, tb = sys.exc_info()
        lineno = traceback.extract_tb(tb)[-1][1]
        lines = code.splitlines()
        errmsg = '{}, \'{}\''.format(detail, lines[lineno - 1])
        ex_factory = e.__class__
    finally :
        if errmsg is not None :
            raise ex_factory(errmsg)
    
    return module

def load_circuit_from_file(filename) :
    with open(filename, 'r') as file :
        qasm = file.read()
    return load_circuit(qasm)
