import sys
from .grammar import OpenQASMGrammar

grammar = OpenQASMGrammar()

def parse(content) :
    lines = content.splitlines()
    # print(len(lines))
    scanned = []
    for line in lines :
        pos = line.find('//')
        if pos != -1 :
            line = line[:pos] + ' ' * (len(line) - pos)
        scanned.append(line)
    content = '\n'.join(scanned)
    
    res = grammar.parse(content)
    # print(res)
    # print(res.is_valid)
    if res.is_valid :
        return res.tree

    # print err line
    from itertools import accumulate
    import bisect

    line_lengths = [len(line) + 1 for line in lines]
    line_offsets = list(accumulate(line_lengths))
    err_lineno = bisect.bisect_right(line_offsets, res.pos)
    errsrc = lines[err_lineno]
    
    raise RuntimeError('parse error at line {}: \'{}\''.format(err_lineno, errsrc))


# Returns properties of a node object as a dictionary:
def node_props(node, children):
    str = node.string if len(node.string) < 1024 else node.string[:1024] + '...(truncated)'
    return {
        'start': node.start,
        'end': node.end,
        'name': node.element.name if hasattr(node.element, 'name') else None,
        'element': node.element.__class__.__name__,
        'string': str,
        'children': children}

# Recursive method to get the children of a node object:
def get_children(children):
    return [node_props(c, get_children(c.children)) for c in children]

# View the parse tree:
def view_parse_tree(tree):
    start = tree.children[0] \
        if tree.children else tree
    return node_props(start, get_children(start.children))


if __name__ == '__main__' :
    import json

    if len(sys.argv) == 1 :
        # use stdin
        file = sys.stdin
        content = file.read()
        doctree = parse(contenxt)
        print(json.dumps(view_parse_tree(doctree), indent=2))
    else :
        print(sys.argv[1:])
        for filename in sys.argv[1:] :
            with open(filename, 'r') as file:
                content = file.read()
                doctree = parse(content)
                print(json.dumps(view_parse_tree(doctree), indent=2))
        
