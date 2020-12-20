from . import mast


def _find_all_positions(line, char):
    result = []
    pos = -1
    while True:
        pos = line.find(char, pos + 1)
        if pos >= 0:
            result.append(pos)
        else:
            break
    return result


def _split_cmd(line):
    spc = _find_all_positions(line, ' ')
    quo = _find_all_positions(line, '"')
    for a, b in zip(quo[::2], quo[1::2]):
        spc = [p for p in spc if p < a or p > b]
    spc = [-1] + spc + [len(line)]
    return [line[a + 1:b] for a, b in zip(spc, spc[1:])]


assert _split_cmd('a b c') == ['a', 'b', 'c']
assert _split_cmd('op add a a 1') == ['op', 'add', 'a', 'a', '1']
assert _split_cmd('cmd') == ['cmd']
assert _split_cmd('"a" "b" "c"') == ['"a"', '"b"', '"c"']
assert _split_cmd('"a a" "b b" "c\nc"') == ['"a a"', '"b b"', '"c\nc"']


def _name_or_literal(line):
    if line == 'null':
        return mast.Literal(None)
    if line == 'true':
        return mast.Literal(True)
    if line == 'false':
        return mast.Literal(False)
    if line[:1] == '"':
        assert line.endswith('"')
        return mast.Literal(line[1:-1])
    try:
        return mast.Literal(int(line))
    except ValueError:
        pass
    try:
        return mast.Literal(float(line))
    except ValueError:
        pass
    return mast.Name(line)


def _parse_cmd(cmd, *args):
    if cmd == 'op':
        return mast.FunctionCall(
            f'op {args[0]}',
            [_name_or_literal(a) for a in args[2:]],
            mast.Name(args[1]),
        )
    elif cmd == 'read':
        return mast.FunctionCall(
            cmd,
            [_name_or_literal(a) for a in args[1:]],
            mast.Name(args[0]),
        )
    elif cmd in ('print', 'printflush', 'end', 'write'):
        return mast.ProcedureCall(
            cmd,
            [_name_or_literal(a) for a in args],
        )
    elif cmd == 'jump':
        return mast.Jump(
            int(args[0]),  # label
            args[1],  # condition
            [_name_or_literal(a) for a in args[2:]],
        )
    else:
        raise ValueError(f'Unknown command {cmd} {" ".join(args)}')


def parse(lines):
    result = []
    for line in lines.splitlines():
        line = line.strip()
        if not line:
            continue
        result.append(_parse_cmd(*_split_cmd(line)))
    return result


p = parse("""
op greaterThan _r1 a 3
jump 4 equal _r1 false
print "Yes"
jump 9 always
op sub _r2 a b
jump 8 equal _r2 false
print "Maybe"
jump 9 always
print "No"
end
""")
print(p)


INST_MAP = {
    'op add': lambda a, b: a + b,
    'op sub': lambda a, b: a - b,
    'op mul': lambda a, b: a * b,
    'op div': lambda a, b: a / b,
    'op idiv': lambda a, b: a // b,
    'op mod': lambda a, b: a % b,
    'op pow': lambda a, b: a ** b,
    'op shl': lambda a, b: a << b,
    'op shr': lambda a, b: a >> b,
    'op or': lambda a, b: a | b,
    'op xor': lambda a, b: a ^ b,
    'op and': lambda a, b: a & b,
    'op land': lambda a, b: int(bool(a and b)),
    'op not': lambda a: ~a,
    'op equal': lambda a, b: a == b,
    'op notEqual': lambda a, b: a != b,
    'op lessThan': lambda a, b: a < b,
    'op lessThanEq': lambda a, b: a <= b,
    'op greaterThan': lambda a, b: a > b,
    'op greaterThanEq': lambda a, b: a >= b,
    'op min': lambda a, b: min(a, b),
    'op max': lambda a, b: max(a, b),
}
JUMP_CONDS = {
    'equal', 'notEqual',
    'lessThan', 'lessThanEq',
    'greaterThan', 'greaterThanEq',
}


def _print_literal_val(value):
    if value is None:
        return 'null'
    elif value is False:
        return '0'
    elif value is True:
        return '1'
    else:
        return str(value)


def execute(instructions, state: dict, max_steps=5000):
    def _eval_arg(arg):
        if isinstance(arg, mast.Literal):
            return arg.value
        elif isinstance(arg, mast.Name):
            assert arg.name is not None
            return state[arg.name]
        else:
            raise TypeError('Unable to evaluate argument')

    inst_map = INST_MAP.copy()
    inst_map['read'] = lambda cell, index: cell[index]

    ptr = 0
    print_buf = []

    for _ in range(max_steps):
        ins = instructions[ptr]
        ptr += 1
        if isinstance(ins, mast.FunctionCall):
            assert ins.result.name is not None
            state[ins.result.name] = inst_map[ins.op](
                *map(_eval_arg, ins.args))
        elif isinstance(ins, mast.ProcedureCall):
            if ins.op == 'print':
                print_buf.append(_print_literal_val(_eval_arg(ins.args[0])))
            elif ins.op == 'printflush':
                pass
            elif ins.op == 'write':
                value, cell, index = *map(_eval_arg, ins.args)
                cell[index] = value
            elif ins.op == 'end':
                break
            else:
                raise ValueError('Unknown procedure')
        elif isinstance(ins, mast.Jump):
            apply = False
            if ins.op == 'always':
                apply = True
            elif ins.op in JUMP_CONDS:
                apply = inst_map[f'op {ins.op}'](*map(_eval_arg, ins.args))
            else:
                raise ValueError('Unknown jump condition')
            if apply:
                ptr = ins.label
        else:
            raise ValueError('Unknown instruction')
        if ptr >= len(instructions):
            break
    else:
        raise OverflowError('Too many steps have been taken')
    return ''.join(print_buf)
