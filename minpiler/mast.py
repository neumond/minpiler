from dataclasses import dataclass
from typing import Any, List, Union


@dataclass
class Literal:
    value: Any  # any scalar

    def dump(self):
        c = self.value
        if c is None:
            return 'null'
        elif c is False:
            return 'false'
        elif c is True:
            return 'true'
        elif isinstance(c, (int, float)):
            return str(c)
        elif isinstance(c, str):
            if '"' in c:
                raise ValueError(
                    f'Quotes are not allowed in literal strings {c!r}')
            return '"' + c.replace('\n', r'\n') + '"'
        else:
            raise ValueError(f'Unsupported constant {c!r}')


@dataclass
class Name:
    # if None must be named automatically
    name: str = None


Value = Union[Literal, Name]
LValue = Union[Name]


@dataclass
class ProcedureCall:
    op: str
    args: List[Value]


@dataclass
class FunctionCall:
    op: str
    args: List[Value]
    result: LValue


@dataclass
class Label:
    pass


@dataclass
class Jump:
    label: Label
    op: str  # condition
    args: List[Value]


Instruction = Union[ProcedureCall, FunctionCall, Label, Jump]


def remove_unnecessary_jumps(instructions):
    remove = set()
    for index, (a, b) in enumerate(zip(instructions, instructions[1:])):
        if isinstance(a, Jump) and isinstance(b, Label):
            if a.label is b:
                remove.add(index)
    return [ins for i, ins in enumerate(instructions) if i not in remove]


def filter_and_index_labels(instructions):
    labels = {}
    program = []
    for ins in instructions:
        if isinstance(ins, Label):
            labels[id(ins)] = len(program)
        else:
            program.append(ins)
    return program, labels


def create_dump_session(resolve_jump):
    from itertools import count

    c = count(start=1)

    def dump_argument(arg: Value):
        if isinstance(arg, Literal):
            return arg.dump()
        elif isinstance(arg, Name):
            if arg.name is None:
                arg.name = f'_r{next(c)}'
            return arg.name
        raise ValueError

    def dump_instruction(ins: Instruction):
        if isinstance(ins, ProcedureCall):
            return [
                ins.op,
                *map(dump_argument, ins.args),
            ]
        elif isinstance(ins, FunctionCall):
            return [
                ins.op,
                dump_argument(ins.result),
                *map(dump_argument, ins.args),
            ]
        elif isinstance(ins, Jump):
            return [
                'jump',
                str(resolve_jump(ins.label)),
                ins.op,
                *map(dump_argument, ins.args),
            ]
        raise ValueError

    return dump_instruction


def dump(instructions):
    instructions = remove_unnecessary_jumps(instructions)
    instructions, labels = filter_and_index_labels(instructions)

    max_jump = -1

    def resolve_jump(label: Label) -> int:
        nonlocal max_jump
        value = labels[id(label)]
        if value > max_jump:
            max_jump = value
        return value

    result = []
    dump_instruction = create_dump_session(resolve_jump)
    for ins in instructions:
        result.append(' '.join(dump_instruction(ins)))

    if max_jump >= len(result):
        assert max_jump == len(result)
        result.append('end')

    return result
