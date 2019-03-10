from . import model

def expand_operator_list(oplist) :
    expanded = list()
    for op in oplist :
        if isinstance(op, model.Clause) :
            child_ops = expand_operator_list(op.ops)
            expanded += child_ops
        elif isinstance(op, model.IfClause) :
            if_clause = model.IfClause(op.refs, op.val)
            child_clause = expand_clauses(op.clause)
            if_clause.set_clause(child_clause)
            expanded.append(if_clause)
        else :
            expanded.append(op)  # FIXME: clone here ?
            
    return expanded


def expand_clauses(clause) :
    expanded = model.Clause()
    expanded.ops = expand_operator_list(clause.ops)
    return expanded
