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
        return mast.Literal(1.0)
    if line == 'false':
        return mast.Literal(0.0)
    if line[:1] == '"':
        assert line.endswith('"')
        return mast.Literal(line[1:-1])
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
    elif cmd in ('set', 'read'):
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
