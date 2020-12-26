import argparse
import ast
import sys

from . import mind, mast, utils


parser = argparse.ArgumentParser()
parser.add_argument(
    'file', type=argparse.FileType('r'))
parser.add_argument(
    '--output', type=argparse.FileType('w'), default=sys.stdout)


def py_to_mind(code):
    program = []
    global_scope = utils.Scope(None)
    for stmt in ast.parse(code).body:
        _, lines = mind.transform_expr(stmt, global_scope)
        program.extend(lines)
    return '\n'.join(mast.dump(program)) + '\n'


def main(in_file, out_file):
    out_file.write(py_to_mind(in_file.read()))


def run():
    args = parser.parse_args()
    main(args.file, args.output)
