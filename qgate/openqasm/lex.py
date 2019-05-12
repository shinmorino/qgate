import ply.lex as lex
import re

literals = ('+', '-', '/', '*', '^', '(', ')', '[', ']', ';', '"', ',', '{', '}', '>')

tokens = [
    'ID',
    'REAL',
    'NMINTEGER',
    'FILENAME',
    'UNARYOP',
    'PI',
    'EQ',
]

reserved = {
    'OPENQASM' : 'OPENQASM', # not in original BNF
    'include' : 'INCLUDE',   # not in original BNF
    'gate' : 'GATE',
    'barrier' : 'BARRIER',
    'reset' : 'RESET',
    'measure' : 'MEASURE',
    'U' : 'U',
    'CX' : 'CX',
    'qreg' : 'QREG',
    'creg' : 'CREG',
    'opaque' : 'OPAQUE',
    'if' : 'IF',
}

tokens += list(reserved.values())

# reserved words starting with a large letter.
t_OPENQASM = 'OPENQASM'
t_U = 'U'
t_CX = 'CX'

t_EQ = '=='
t_FILENAME = r'\"[A-Za-z0-9_\.]+\"'

def t_REAL(t) :
    r'[0-9]+(\.[0-9]*)?([eE][-+]?[0-9]+)?'
    # hook an action to mis-parsed tokens.
    if re.match(r'[1-9]+[0-9]*(?!\.)|0', t.value) is not None :
        t.type = 'NMINTEGER'
    return t

unaryop_tokens = [ 'sin', 'cos', 'tan', 'exp', 'ln', 'sqrt' ]

def t_ID(t) :
    r'[a-z][A-Za-z0-9_]*'
    # hook an action to mis-parsed tokens.
    t.type = reserved.get(t.value,'ID')
    if t.value == 'pi' : # 'PI'
        t.type = 'PI'
    elif t.value in unaryop_tokens : # 'UNARYOP'
        t.type = 'UNARYOP'
    return t

def t_comment(t) :
    r'//[^\n]*\n'
    t.lexer.lineno += 1

def t_newline(t):
    r'\n+'
    t.lexer.lineno += len(t.value)
    
t_ignore = ' \t'

def t_error(t):
    err = SyntaxError('Syntax Error')
    err.lineno, err.pos = t.lineno, t.lexpos
    raise err

# instancing lexer in module level.
lexer = None
def reset(**params) :
    global lexer
    lexer = lex.lex(**params)

if __name__ == '__main__' :
    qasm = None # for debug

    reset()
    
    import sys
    if qasm is not None :
        content = qasm
    elif len(sys.argv) == 1 :
        # use stdin
        file = sys.stdin
        content = file.read()
    else :
        filename = sys.argv[1]
        with open(filename, 'r') as file:
            content = file.read()

    # normalize line ends
    lines = content.splitlines()
    content = '\n'.join(lines)

    # parse
    try :
        lexer.input(content)
        for token in lexer :
            print(token)
    except SyntaxError as e :
        print(e.lineno, e.pos)
