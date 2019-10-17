import ast
import os
import shutil

class PositionedText:
    def __init__(self, text):
        self.lines = text.splitlines(True)
        self.offsets = [0] * len(self.lines)

    def get(self):
        return ''.join(self.lines)

    def replace(self, old, new, lineno, col_offset):
        line = self.lines[lineno]
        replace_begin = col_offset + self.offsets[lineno]
        replace_end = replace_begin + len(old)
        current = line[replace_begin:replace_end]
        assert current == old, 'internal error.' # current token must match the old token.
        line = line[:col_offset] + new + line[replace_end:]
        self.lines[lineno] = line
        self.offsets[lineno] += len(new) - len(old)

def replace(source, filename):
    tokenmap = {'release_qreg': 'ReleaseQreg',
                'measure': 'Measure', 'reset': 'Reset',
                'ctrl': 'Ctrl', 'controlled': 'Controlled',
                'prob': 'Prob', 'barrier': 'Barrier',
                'if_': 'If'}

    tree = ast.parse(source, filename)
    ast.fix_missing_locations(tree)
    # print(ast.dump(tree, True, True))

    replacements = list()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and hasattr(node.func, 'id'):
            name = node.func.id
            if name in tokenmap.keys():
                new = tokenmap[name]
                token = (name, new, node.lineno - 1, node.col_offset)
                replacements.append(token)
    replacements.sort(key = lambda replacement: replacement[3])

    text = PositionedText(source)
    for replacement in replacements:
        text.replace(*replacement)
    return text.get()

def process_file(filename, overwrite):
    import shutil
    with open(filename, 'r') as f:
        source = f.read()
    updated = replace(source, filename)
    updated_filename = filename + '.updated'
    with open(updated_filename, 'w') as f:
        f.write(updated)
    if overwrite:
        os.remove(filename)
    else:
        shutil.move(filename, filename + '.org')
    shutil.move(updated_filename, filename)

if __name__ == '__main__':
    import sys
    args = sys.argv[1:]

    overwrite = False
    if args[0] == '-o' or args[0] == '--overwrite':
        overwrite = True
        args = args[1:]

    for filename in args:
        try:
            process_file(filename, overwrite)
        except Exception as ex:
            print(ex)
