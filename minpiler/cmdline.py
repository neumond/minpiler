import argparse
import ast
import sys

from . import mind, mast


parser = argparse.ArgumentParser()
parser.add_argument(
    'file', type=argparse.FileType('r'))
parser.add_argument(
    '--output', type=argparse.FileType('w'), default=sys.stdout)


def py_to_mind(code):
    program = []
    for stmt in ast.parse(code).body:
        program.extend(mind.transform_statement(stmt))
    return '\n'.join(mast.dump(program)) + '\n'


def main(in_file, out_file):
    out_file.write(py_to_mind(in_file.read()))


def run():
    args = parser.parse_args()
    main(args.file, args.output)
