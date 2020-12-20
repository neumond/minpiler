import argparse
import ast
import sys

from . import mind, mast


parser = argparse.ArgumentParser()
parser.add_argument(
    'file', type=argparse.FileType('r'))
parser.add_argument(
    '--output', type=argparse.FileType('w'), default=sys.stdout)


def main(in_file, out_file):
    code = in_file.read()

    program = []
    for stmt in ast.parse(code).body:
        program.extend(mind.transform_statement(stmt))
    for line in mast.dump(program):
        out_file.write(line)
        out_file.write('\n')


def run():
    args = parser.parse_args()
    main(args.file, args.output)
