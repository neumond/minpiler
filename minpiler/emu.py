import math
import random

from . import mast


class Var:
    __slots__ = ('value', )

    def __init__(self, val):
        self.value = val

    @property
    def is_object(self) -> bool:
        v = self.value
        if v is None:
            return True
        if isinstance(v, (int, float)):
            return False
        return True

    @property
    def num_value(self) -> float:
        v = self.value
        if v is None:
            return 0.0
        if v is True:
            return 1.0
        if v is False:
            return 0.0
        if isinstance(v, int):
            return float(v)
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return 0.0
            return v
        return 1.0

    @property
    def int_value(self) -> int:
        return int(self.num_value)

    @property
    def printable(self) -> str:
        v = self.value
        if v is None:
            return 'null'
        elif isinstance(v, float):
            return str(v).rstrip('0').rstrip('.')
        elif isinstance(v, dict):
            return 'cell'
        else:
            return str(v)


def require_type(val, type_, fn, fallback):
    if isinstance(val, type_):
        return val
    return fallback


def _eq_func(a, b):
    if a.is_object and b.is_object:
        return a.value == b.value
    return a.num_value == b.num_value


def _read_func(cell, index):
    if not isinstance(cell.value, dict):
        return 0.0
    return cell.value.get(index.int_value, 0.0)


def _dst_func(a, b):
    a, b = a.num_value, b.num_value
    return math.sqrt(a * a + b * b)


_DTR = math.pi / 180
_RTD = 180 / math.pi


INST_MAP = {
    'op add': lambda a, b: a.num_value + b.num_value,
    'op sub': lambda a, b: a.num_value - b.num_value,
    'op mul': lambda a, b: a.num_value * b.num_value,
    'op div': lambda a, b: a.num_value / b.num_value,
    'op idiv': lambda a, b: a.num_value // b.num_value,
    'op mod': lambda a, b: a.num_value % b.num_value,
    'op pow': lambda a, b: a.num_value ** b.num_value,
    'op shl': lambda a, b: float(a.int_value << b.int_value),
    'op shr': lambda a, b: float(a.int_value >> b.int_value),
    'op or': lambda a, b: float(a.int_value | b.int_value),
    'op xor': lambda a, b: float(a.int_value ^ b.int_value),
    'op and': lambda a, b: float(a.int_value & b.int_value),
    'op land': lambda a, b: float(bool(a.num_value and b.num_value)),
    'op not': lambda a: float(~a.int_value),
    'op equal': lambda a, b: float(_eq_func(a, b)),
    'op notEqual': lambda a, b: float(not _eq_func(a, b)),
    'op lessThan': lambda a, b: float(a.num_value < b.num_value),
    'op lessThanEq': lambda a, b: float(a.num_value <= b.num_value),
    'op greaterThan': lambda a, b: float(a.num_value > b.num_value),
    'op greaterThanEq': lambda a, b: float(a.num_value >= b.num_value),
    'op min': lambda a, b: min(a.num_value, b.num_value),
    'op max': lambda a, b: max(a.num_value, b.num_value),
    'op atan2': lambda a, b: _RTD * math.atan2(a.num_value, b.num_value),
    'op dst': _dst_func,
    # TODO: op noise
    'op abs': lambda a: abs(a.num_value),
    'op log': lambda a: math.log(a.num_value),
    'op log10': lambda a: math.log10(a.num_value),
    'op sin': lambda a: math.sin(_DTR * a.num_value),
    'op cos': lambda a: math.cos(_DTR * a.num_value),
    'op tan': lambda a: math.tan(_DTR * a.num_value),
    'op floor': lambda a: float(math.floor(a.num_value)),
    'op ceil': lambda a: float(math.ceil(a.num_value)),
    'op sqrt': lambda a: math.sqrt(a.num_value),
    'op rand': lambda a: random.random() * a.num_value,
    'set': lambda a: a.value,
    'read': _read_func,
}
JUMP_CONDS = {
    'equal', 'notEqual',
    'lessThan', 'lessThanEq',
    'greaterThan', 'greaterThanEq',
}


class OutputBuffer:
    def __init__(self):
        self._print_buf = []

    def print(self, text):
        self._print_buf.append(text)

    def printflush(self):
        r = ''.join(self._print_buf)
        self._print_buf.clear()
        return r


def execute_fn_call(ins, fn, args):
    try:
        return fn(*args)
    except (ZeroDivisionError, OverflowError):
        return 0.0


def execute_proc_call(ins, args, buf):
    if ins.op == 'print':
        buf.print(args[0].printable)
    elif ins.op == 'printflush':
        pass
    elif ins.op == 'write':
        value = args[0].num_value
        cell = args[1].value
        index = args[2].num_value
        cell[index] = value
    else:
        raise ValueError('Unknown procedure')


def create_arg_evaluator(state):
    def _eval_arg(arg):
        if isinstance(arg, mast.Literal):
            return Var(arg.value)
        elif isinstance(arg, mast.Name):
            assert arg.name is not None
            return Var(state.get(arg.name, None))
        else:
            raise TypeError('Unable to evaluate argument')
    return _eval_arg


def execute(instructions, state: dict, max_steps=5000):
    _eval_arg = create_arg_evaluator(state)

    inst_map = INST_MAP.copy()

    ptr = 0
    buf = OutputBuffer()

    for _ in range(max_steps):
        ins = instructions[ptr]
        ptr += 1
        if isinstance(ins, mast.FunctionCall):
            state[ins.result.name] = execute_fn_call(
                ins,
                inst_map[ins.op],
                list(map(_eval_arg, ins.args)),
            )
        elif isinstance(ins, mast.ProcedureCall):
            if ins.op == 'end':
                break
            else:
                execute_proc_call(
                    ins,
                    list(map(_eval_arg, ins.args)),
                    buf,
                )
        elif isinstance(ins, mast.Jump):
            apply = False
            if ins.op == 'always':
                apply = True
            elif ins.op in JUMP_CONDS:
                fn = inst_map[f'op {ins.op}']
                apply = fn(*map(_eval_arg, ins.args))
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
    return buf.printflush()
