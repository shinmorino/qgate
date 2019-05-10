import ply.yacc as yacc

from . import lex
from .lex import tokens, literals

precedence = (
    ('left', '+', '-'),
    ('left', '*', '/'),
    ('left', '^'),
#    ('right', '-'),  # warning: already considered.
    ('left', 'ID'),
    ('left', 'U'),
    ('left', 'CX'),
    ('left', 'BARRIER'),
    ('left', 'OPENQASM', 'INCLUDE'),
)

# The original simplified BNF does not have header and include list.
def p_mainprogram(p) :
    ''' mainprogram : header_or_empty includelist_or_empty program_begin program program_end '''

# empty action to trigger analyzer.
def p_program_begin(p) :
    ''' program_begin : '''
    this.analyzer.open_program()

# empty action to trigger analyzer.
def p_program_end(p) :
    ''' program_end : '''
    this.analyzer.close_program()

# added to the oroginal BNF.
def p_header_or_empty(p) :
    ''' header_or_empty : OPENQASM REAL ';' 
                        | '''
    pass

# added to the oroginal BNF.
def p_includelist_or_empty(p) :
    ''' includelist_or_empty : includelist
                             | '''
    pass

# added to the oroginal BNF.
def p_includelist(p) :
    ''' includelist : includelist include
                    | include '''
    pass

# added to the oroginal BNF.
def p_include(p) :
    ''' include : INCLUDE FILENAME ';'  '''
    if p[2] == '"qelib1.inc"' :
        p[0] == True
    else :
        raise NotImplementedError('include is not implemented.')

# modified to allow an empty program.
def p_program(p) :
    ''' program : statement
                | program statement
                | '''
    pass

# empty goplist is added.
def p_statement(p) :
    ''' statement : decl
                  | gatedecl goplist '}'
                  | gatedecl  '}'
                  | OPAQUE ID idlist ';'
                  | OPAQUE ID '(' ')' idlist ';'
                  | OPAQUE ID '(' exp ')' idlist ';'
                  | qop
                  | IF '(' ID EQ NMINTEGER ')' open_if qop
                  | BARRIER anylist ';' '''
    if p.slice[1].type == 'OPAQUE' :
        raise NotImplementedError('opaque is not implemented.')  # FIXME: error ?
    elif p.slice[1].type == 'gatedecl' :
        raise NotImplementedError('gatedecl not implemented')
    elif p.slice[1].type == 'BARRIER' :
        this.analyzer.barrier(p[2])
    elif p.slice[1].type == 'IF' :
        this.analyzer.close_if()
    elif p[1] != True :
        print(p[1])
        raise RuntimeError('unprocessed statement found.')

# empty action to trigger analyzer.
def p_open_if(p) :
    ''' open_if : '''
    this.analyzer.open_if(p[-4], int(p[-2]))
    
def p_decl(p) :
    ''' decl : QREG ID '[' NMINTEGER ']' ';'
             | CREG ID '[' NMINTEGER ']' ';' '''
    if p[1] == 'qreg' or p[1] == 'creg' :
        this.analyzer.decl(p[1], p[2], p[4])
        p[0] = True
    else :
        parser_broken()

def p_gatedecl(p) :
    ''' gatedecl : GATE ID idlist '{'
                 | GATE ID '(' ')' idlist '{'
                 | GATE ID '(' idlist ')' idlist '{' '''
    pass

def p_goplist(p) :
    ''' goplist : uop
                | BARRIER idlist ';'
                | goplist uop
                | goplist BARRIER idlist ';' '''
    pass

def p_qop(p) :
    ''' qop : uop
            | MEASURE argument '-' '>' argument ';'
            | RESET argument ';' '''
    if p[1] == 'measure' :
        this.analyzer.measure(p[2], p[5])
        p[0] = True
    elif p[1] == 'reset' :
        this.analyzer.reset(p[2])
        p[0] = True
    else :
        p[0] = p[1]

def p_uop(p) :
    ''' uop : U '(' explist ')' argument ';' %prec U
            | CX argument ',' argument ';' %prec CX
            | ID anylist ';' %prec ID
            | ID '(' ')' anylist ';' %prec ID
            | ID '(' explist ')' anylist ';' %prec ID ''' 
    if p[1] == 'U' :
        this.analyzer.U_gate(p[3], p[5])
    elif p[1] == 'CX' :
        this.analyzer.CX_gate(p[2], p[4])
    elif p.slice[1].type == 'ID' :
        id_ = p[1]
        if len(p) == 4 :
            explist = None
            anylist = p[2]
        elif len(p) == 6 :
            explist = None
            anylist = p[4]
        else : # len(p) == 7
            explist = p[3]
            anylist = p[5]
        this.analyzer.id_gate(id_, explist, anylist)
    else :
        parser_broken()
    p[0] = True

def p_anylist(p) :
    ''' anylist : idlist
                | mixedlist '''
    p[0] = p[1]

def p_idlist(p) :
    ''' idlist : ID
               | idlist ',' ID '''
    if len(p) == 2 :
        p[0] = [p[1]]
    else :
        p[0] = p[1]
        p[0].append(p[3])

def p_mixedlist(p) :
    ''' mixedlist : ID '[' NMINTEGER ']'
                  | mixedlist ',' ID
                  | mixedlist ',' ID '[' NMINTEGER ']'
                  | idlist ',' ID '[' NMINTEGER ']' '''
    if p.slice[1].type == 'ID' :
        # indexed id.
        idxid = (p[1], int(p[3]))
        p[0] = [idxid]
    elif isinstance(p[1], list) :
        if len(p) == 4 :
            # mixedlist += ID
            p[0] = p[1]
            p[0].append((p[3], ))
        else : # len(p) == 7
            p[0] = p[1]
            p[0].append((p[3], int(p[5])))
    else :
        parser_broken()

def p_argument(p) :
    ''' argument : ID
                 | ID '[' NMINTEGER ']'  '''
    if len(p) == 2 :
        p[0] = (p[1], )
    elif len(p) == 5 :
        p[0] = (p[1], int(p[3]))
    else :
        parser_broken()

# The rule of '| exp' is added to terminate list.
def p_explist(p) :
    ''' explist : explist ',' exp
                | exp '''
    if len(p) == 2 :
        p[0] = [ p[1] ]
    elif len(p) == 4 :
        p[0] = p[1]
        p[0].append(p[3])
    else :
        parser_broken()

def p_exp(p) :
    ''' exp : REAL
            | NMINTEGER
            | PI
            | ID
            | exp '+' exp %prec '+'
            | exp '-' exp %prec '-'
            | exp '*' exp %prec '*'
            | exp '/' exp %prec '/'
            | '-' exp %prec '-'
            | exp '^' exp
            | '(' exp ')'
            | UNARYOP '(' exp ')'
    '''
    if p.slice[1].type == 'PI' :
        p[1] = 'math.pi'
    elif p.slice[1].type == 'UNARYOP' :
        p[1] = 'math.' + p[1]
    p[0] = ' '.join(p[1:])

def p_error(p) :
    e = SyntaxError()
    e.lineno, e.pos = p.lineno, p.lexpos
    raise e

import sys
this = sys.modules[__name__]

def parser_broken() :
    assert False, 'Unexpected action, parser may be broken.'

def parse(content, analyzer) :
    lex.reset()
    this.analyzer = analyzer
    # normalize line ends
    lines = content.splitlines()
    content = '\n'.join(lines)
    # parser = yacc.yacc(debug = True, write_tables = False)
    parser = yacc.yacc()
    errmsg = None
    try :
        parser.parse(content)
    except SyntaxError as e :
        # create char offset for lines.
        from itertools import accumulate
        line_lengths = [len(line) + 1 for line in lines]
        line_offsets = list(accumulate(line_lengths))
        # error message
        lineidx_begin = max(e.lineno - 2, 0)
        lineidx_end = min(e.lineno, len(line_offsets))
        linepos = e.pos
        if e.lineno != 1 :
            linepos -= line_offsets[e.lineno - 2]
        err_lines = ' '.join(lines[lineidx_begin:lineidx_end])
        errmsg = 'invalid syntax, {}:{}:\'{}\''.format(e.lineno, linepos + 1, err_lines)
    finally :
        if errmsg is not None :
            raise SyntaxError(errmsg)

if __name__ == '__main__' :
    import sys
    if len(sys.argv) == 1 :
        # use stdin
        file = sys.stdin
        content = file.read()
    else :
        filename = sys.argv[1]
        with open(filename, 'r') as file:
            content = file.read()

    parse(content)
