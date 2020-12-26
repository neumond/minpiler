import argparse
import ast
import sys

from . import mind, mast, utils, optimizer


parser = argparse.ArgumentParser()
parser.add_argument(
    'file', type=argparse.FileType('r'))
parser.add_argument(
    '-o', '--output', type=argparse.FileType('w'), default=sys.stdout)
parser.add_argument('-c', '--clip', action='store_true')


def py_to_mind(code):
    program = []
    global_scope = utils.Scope(None)
    for stmt in ast.parse(code).body:
        _, lines = mind.transform_expr(stmt, global_scope)
        program.extend(lines)
    program = optimizer.optimize(program)
    return '\n'.join(mast.dump(program)) + '\n'


def main(in_file, out_file, use_clip):
    code = py_to_mind(in_file.read())
    if use_clip:
        try:
            import pyperclip
        except ImportError:
            print('Install pyperclip to use --clip option', file=sys.stderr)
            exit(1)
        pyperclip.copy(code)
        n = code.count("\n") + 1
        print(f'Copied {n} lines to clipboard')
    else:
        out_file.write(code)


def run():
    args = parser.parse_args()
    main(args.file, args.output, args.clip)
