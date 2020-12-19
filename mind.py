import ast
import string
from dataclasses import dataclass
from itertools import count, islice
from typing import Any, Callable, Iterator

import mast


def label_allocator():
    for i in count(start=1):
        yield f'<label:{i}>'


def register_allocator():
    for i in count(start=1):
        yield f'<reg:{i}>'


def name_allocator(letters):
    ln = len(letters)
    stack = []
    while True:
        pos = 0
        while True:
            if pos >= len(stack):
                stack.append(0)
                break
            stack[pos] += 1
            if stack[pos] >= ln:
                stack[pos] = 0
                pos += 1
            else:
                break
        yield ''.join(letters[idx] for idx in reversed(stack))


assert list(islice(name_allocator('ab'), 10)) == [
    'a', 'b', 'aa', 'ab', 'ba',
    'bb', 'aaa', 'aab', 'aba', 'abb',
]
assert list(islice(name_allocator('a'), 5)) == [
    'a', 'aa', 'aaa', 'aaaa', 'aaaaa',
]
assert list(islice(name_allocator('abcd'), 5)) == [
    'a', 'b', 'c', 'd', 'aa',
]


class RegisterAllocatorContext:
    def __init__(self, allocator):
        self._allocator = allocator

    def allocate(self):
        name = self._allocator.allocate()
        self._names.add(name)
        return name

    def free(self, name):
        self._names.remove(name)
        self._allocator.free(name)

    def __enter__(self):
        self._names = set()
        return self

    def __exit__(self, *args):
        for name in self._names:
            self._allocator.free(name)
        self._names = None


class RegisterAllocator:
    def __init__(self, pre_busy=()):
        self._busy = set(pre_busy)
        self._free = set()
        self._ngen = name_allocator(string.ascii_lowercase)

    def context(self):
        return RegisterAllocatorContext(self)

    def allocate(self):
        try:
            name = self._free.pop()
        except KeyError:
            while True:
                name = '_' + next(self._ngen)
                if (name not in self._busy) and (name not in self._free):
                    break
        self._busy.add(name)
        return name

    def free(self, name):
        self._busy.remove(name)
        self._free.add(name)


def remove_unnecessary_jumps(lines):
    remove_lines = set()
    for a_index, (a, b) in enumerate(zip(lines, lines[1:])):
        if not (a.startswith('jump ') and b.startswith('label ')):
            continue
        _, jmp_label, _ = a.split(' ', 2)
        _, target_label = b.split(' ', 1)
        if jmp_label == target_label:
            remove_lines.add(a_index)
    return [line for i, line in enumerate(lines) if i not in remove_lines]


def optimize(lines):
    lines = remove_unnecessary_jumps(lines)
    return lines


def link_program(lines):
    labels = {}
    program = []
    for line in lines:
        if line.startswith('label '):
            labels[line.split(' ', 1)[1]] = len(program)
        else:
            program.append(line)

    for index, line in enumerate(program):
        if line.startswith('jump '):
            _, label, cmd = line.split(' ', 2)
            program[index] = f'jump {labels[label]} {cmd}'

    if labels and max(labels.values()) >= len(program):
        program.append('end')

    return program


def transform_constant(c):
    assert isinstance(c, ast.Constant), f'{c}'
    c = c.value
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


BIN_OP_MAP = {
    ast.Add: 'add',
    ast.Sub: 'sub',
    ast.Mult: 'mul',
    ast.Div: 'div',
    ast.FloorDiv: 'idiv',
    ast.Mod: 'mod',
    ast.Pow: 'pow',
    ast.LShift: 'shl',
    ast.RShift: 'shr',
    ast.BitOr: 'or',
    ast.BitXor: 'xor',
    ast.BitAnd: 'and',
    # MatMult
}

COND_OP_MAP = {
    ast.Eq: 'equal',
    ast.NotEq: 'notEqual',
    ast.Lt: 'lessThan',
    ast.LtE: 'lessThanEq',
    ast.Gt: 'greaterThan',
    ast.GtE: 'greaterThanEq',
}

BOOL_OP_MAP = {  # op, shortcut_condition
    ast.And: ('land', mast.Literal(False)),
    ast.Or: ('or', mast.Literal(True)),
}

UNARY_OP_MAP = {
    ast.Invert: lambda val, result: mast.FunctionCall(
        'op not', [val], result),
    ast.Not: lambda val, result: mast.FunctionCall(
        'op equal', [mast.Literal(0), val], result),
    ast.UAdd: lambda val, result: mast.FunctionCall(
        'op add', [mast.Literal(0), val], result),
    ast.USub: lambda val, result: mast.FunctionCall(
        'op sub', [mast.Literal(0), val], result),
}


def get_type_map(map, item, desc):
    if type(item) not in map:
        raise ValueError(f'Unsupported {desc} {item}')
    return map[type(item)]


@dataclass
class BaseExpressionHandler:
    expr: Any
    trec: Callable
    alloc_result_name: Callable
    alloc_temp_name: Callable
    label_allocator: Iterator
    _result_name: str = None

    # AST_CLASS = ast.Xxx

    def dev_dump(self):
        print(ast.dump(self.expr))

    @property
    def result_name(self):
        if self._result_name is None:
            self._result_name = self.alloc_result_name()
        return self._result_name

    def handle(self):
        raise NotImplementedError


class ConstantHandler(BaseExpressionHandler):
    AST_CLASS = ast.Constant

    def handle(self):
        return mast.Literal(self.expr.value), []


class NameHandler(BaseExpressionHandler):
    AST_CLASS = ast.Name

    def handle(self):
        return mast.Name(self.expr.id), []


class SubscriptHandler(BaseExpressionHandler):
    AST_CLASS = ast.Subscript

    def handle(self):
        # memory cell access
        assert isinstance(self.expr.slice, ast.Index)
        array_val, array_pre = self.trec(self.expr.value)
        index_val, index_pre = self.trec(self.expr.slice.value)
        return self.result_name, [
            *array_pre,
            *index_pre,
            mast.FunctionCall(
                'read', [array_val, index_val], self.result_name),
        ]


class UnaryOpHandler(BaseExpressionHandler):
    AST_CLASS = ast.UnaryOp

    def handle(self):
        factory = get_type_map(UNARY_OP_MAP, self.expr.op, 'UnaryOp')
        val, pre = self.trec(self.expr.operand)
        return self.result_name, [
            *pre,
            factory(val, self.result_name),
        ]


class BinOpHandler(BaseExpressionHandler):
    AST_CLASS = ast.BinOp

    def handle(self):
        op = get_type_map(BIN_OP_MAP, self.expr.op, 'BinOp')
        left_val, left_pre = self.trec(self.expr.left)
        right_val, right_pre = self.trec(self.expr.right)
        return self.result_name, [
            *left_pre,
            *right_pre,
            mast.FunctionCall(
                f'op {op}', [left_val, right_val], self.result_name),
        ]


class CompareHandler(BaseExpressionHandler):
    AST_CLASS = ast.Compare

    def handle(self):
        end_label = mast.Label()
        a_val, pre = self.trec(self.expr.left)

        for op, comparator in zip(self.expr.ops, self.expr.comparators):
            op = get_type_map(COND_OP_MAP, op, 'Compare')
            b_val, b_pre = self.trec(comparator)
            pre.extend(b_pre)
            pre.append(mast.FunctionCall(
                f'op {op}', [a_val, b_val], self.result_name))
            pre.append(mast.Jump(
                end_label, 'equal', [self.result_name, mast.Literal(False)]))
            a_val = b_val

        pre.append(end_label)
        return self.result_name, pre


class BoolOpHandler(BaseExpressionHandler):
    AST_CLASS = ast.BoolOp

    def handle(self):
        op, shortcut_condition = get_type_map(
            BOOL_OP_MAP, self.expr.op, 'BoolOp')

        end_label = mast.Label()
        val, pre = self.trec(self.expr.values[0])
        pre.append(mast.FunctionCall('set', [val], self.result_name))

        bool_value = mast.Name()

        for value in self.expr.values[1:]:
            val, b_pre = self.trec(value)
            pre.extend(b_pre)
            pre.append(mast.FunctionCall(
                f'op {op}', [self.result_name, val], bool_value))
            pre.append(mast.Jump(
                end_label, 'equal', [bool_value, shortcut_condition]))
            pre.append(mast.FunctionCall(
                'set', [val], self.result_name))

        pre.append(end_label)
        return self.result_name, pre


class CallHandler(BaseExpressionHandler):
    AST_CLASS = ast.Call

    def func_min(self, a, b):
        # TODO: support multiple values
        return [mast.FunctionCall('op min', [a, b], self.result_name)]

    def func_max(self, a, b):
        # TODO: support multiple values
        return [mast.FunctionCall('op max', [a, b], self.result_name)]

    def func_print(self, *args):
        return [mast.ProcedureCall('print', [arg]) for arg in args]

    def func_printflush(self, target):
        return [mast.ProcedureCall('printflush', [target])]

    def func_exit(self):
        return [mast.ProcedureCall('end', [])]

    def handle(self):
        if not isinstance(self.expr.func, ast.Name):
            raise ValueError(
                'Expressions resulting in functions are not allowed, '
                'only direct calls of named functions: func(1, 2, 3)'
            )
        fname = self.expr.func.id
        method = getattr(self, 'func_' + fname, None)
        if method is None:
            raise ValueError(f'Unknown function name {fname}')

        if self.expr.keywords:
            raise ValueError('Keyword arguments are not supported')

        arg_vals = []
        result_pre = []
        for arg in self.expr.args:
            val, pre = self.trec(arg)
            arg_vals.append(val)
            result_pre.extend(pre)

        result_pre.extend(method(*arg_vals))

        if self._result_name is None:
            return mast.Literal(None), result_pre
        else:
            return self.result_name, result_pre


class AttributeHandler(BaseExpressionHandler):
    AST_CLASS = ast.Attribute

    def obj_Material(self, attr):
        return mast.Name(f'@{attr}'), []

    def handle(self):
        if not isinstance(self.expr.value, ast.Name):
            raise ValueError(
                'Expressions are not allowed before attribute access, '
                'use names of objects directly: Material.copper'
            )
        obj_name = self.expr.value.id
        method = getattr(self, 'obj_' + obj_name, None)
        if method is None:
            raise ValueError(f'Unknown object name {obj_name}')
        return method(self.expr.attr)


AST_NODE_MAP = {
    subcls.AST_CLASS: subcls
    for subcls in BaseExpressionHandler.__subclasses__()
}


def transform_expr(
    expr, register_allocator, alloc_result_name, label_allocator,
):
    with register_allocator.context() as reg:

        def trec(expr):
            return transform_expr(
                expr, register_allocator, mast.Name, label_allocator)

        if type(expr) not in AST_NODE_MAP:
            raise ValueError(f'Unsupported expression {expr}')

        return AST_NODE_MAP[type(expr)](
            expr, trec,
            alloc_result_name, mast.Name, label_allocator).handle()


def test_transform_expr(code):
    assert len(ast.parse(code).body) == 1
    expr = ast.parse(code).body[0]
    assert isinstance(expr, ast.Expr)
    val, lines = transform_expr(
        expr.value,
        RegisterAllocator(),
        lambda: mast.Name('result'),
        label_allocator(),
    )
    # lines = optimize(lines)
    print('-----')
    lines.append(mast.ProcedureCall('print', [val]))
    for line in mast.dump(lines):
        print(line)
    # print('print', val)


# test_transform_expr('2+2')
# test_transform_expr('2 + 2 * 2 + 8 + 6 * 9 * 3')
# test_transform_expr('-5')
# test_transform_expr('cell["kek"]')
# test_transform_expr('max(min(2, 8), 3 + 3)')
# test_transform_expr('print(1, 2 + 7, 3, print(), "lol")')
# test_transform_expr('printflush(message1)')
# test_transform_expr('Material.copper')
# test_transform_expr('exit()')
# test_transform_expr('1 >= a > 3')
test_transform_expr('True and True or False and 3')

# set _a true
# op land _b _a true
# label <label:2>
# set result _a
# set _c false
# op land _d _c 3
# label <label:3>
# op or result result _c
# label <label:1>
# print result

exit()


@dataclass
class BaseStatementHandler:
    stmt: Any
    register_allocator: RegisterAllocator
    label_allocator: Iterator

    # AST_CLASS = ast.Xxx

    def dev_dump(self):
        print(ast.dump(self.stmt))

    def eval_expr(self, expr, alloc_result_name):
        return transform_expr(
            expr, self.register_allocator, alloc_result_name,
            self.label_allocator)

    def handle(self):
        raise NotImplementedError


class ExprStatementHandler(BaseStatementHandler):
    AST_CLASS = ast.Expr

    def handle(self):
        with self.register_allocator.context() as reg:
            retval, pre = self.eval_expr(self.stmt.value, reg.allocate)
        return pre


class AssignStatementHandler(BaseStatementHandler):
    AST_CLASS = ast.Assign

    def named_assign(self, target, value):
        retval, pre = self.eval_expr(value, lambda: target.id)
        if retval != target.id:
            pre.append(f'set {target.id} {retval}')
        return pre

    def memory_assign(self, target, value):
        if not isinstance(target.value, ast.Name):
            raise ValueError(f'Unsupported assignment target {target}')
        assert isinstance(target.slice, ast.Index)
        with self.register_allocator.context() as reg:
            index_val, index_pre = self.eval_expr(
                target.slice.value, reg.allocate)
            value_val, value_pre = self.eval_expr(
                value, reg.allocate)
        return [
            *index_pre,
            *value_pre,
            f'write {value_val} {target.value.id} {index_val}',
        ]

    TARGET_MAP = {
        ast.Name: named_assign,
        ast.Subscript: memory_assign,
    }

    def handle(self):
        if len(self.stmt.targets) != 1:
            raise ValueError(
                'Only single target can be used in assignment: a = 3')
        target = self.stmt.targets[0]
        if type(target) not in self.TARGET_MAP:
            raise ValueError(f'Unsupported assignment target {target}')
        method = self.TARGET_MAP[type(target)]
        return method(self, target, self.stmt.value)


class AugAssignStatementHandler(BaseStatementHandler):
    AST_CLASS = ast.AugAssign

    def named_assign(self, target, op, operand):
        with self.register_allocator.context() as reg:
            operand_val, pre = self.eval_expr(operand, reg.allocate)
            pre.append(f'op {op} {target.id} {target.id} {operand_val}')
            return pre

    def memory_assign(self, target, op, operand):
        if not isinstance(target.value, ast.Name):
            raise ValueError(f'Unsupported assignment target {target}')
        assert isinstance(target.slice, ast.Index)
        with self.register_allocator.context() as reg:
            index_val, index_pre = self.eval_expr(
                target.slice.value, reg.allocate)
            operand_val, operand_pre = self.eval_expr(
                operand, reg.allocate)
            op_output = reg.allocate()
            return [
                *index_pre,
                *operand_pre,
                f'read {op_output} {target.value.id} {index_val}',
                f'op {op} {op_output} {op_output} {operand_val}',
                f'write {op_output} {target.value.id} {index_val}',
            ]

    TARGET_MAP = {
        ast.Name: named_assign,
        ast.Subscript: memory_assign,
    }

    def handle(self):
        target = self.stmt.target
        if type(target) not in self.TARGET_MAP:
            raise ValueError(f'Unsupported assignment target {target}')
        method = self.TARGET_MAP[type(target)]
        if type(self.stmt.op) not in BIN_OP_MAP:
            raise ValueError(f'Unsupported BinOp {self.stmt.op}')
        return method(
            self, target, BIN_OP_MAP[type(self.stmt.op)], self.stmt.value)


class IfStatementHandler(BaseStatementHandler):
    AST_CLASS = ast.If

    def handle(self):
        if self.stmt.orelse:
            else_label = next(self.label_allocator)
        end_label = next(self.label_allocator)
        if not self.stmt.orelse:
            else_label = end_label

        result = []

        with self.register_allocator.context() as reg:
            test_val, test_pre = self.eval_expr(
                self.stmt.test, reg.allocate)
            result.extend(test_pre)
            result.append(f'jump {else_label} equal {test_val} false')

        for stmt in self.stmt.body:
            result.extend(transform_statement(
                stmt, self.register_allocator, self.label_allocator))

        if self.stmt.orelse:
            result.append(f'jump {end_label} always')
            result.append(f'label {else_label}')

            for stmt in self.stmt.orelse:
                result.extend(transform_statement(
                    stmt, self.register_allocator, self.label_allocator))

        result.append(f'label {end_label}')
        return result


AST_STATEMENT_MAP = {
    subcls.AST_CLASS: subcls
    for subcls in BaseStatementHandler.__subclasses__()
}


def transform_statement(stmt, register_allocator, label_allocator):
    if type(stmt) not in AST_STATEMENT_MAP:
        raise ValueError(f'Unsupported statement {stmt}')

    return AST_STATEMENT_MAP[type(stmt)](
        stmt, register_allocator, label_allocator).handle()


def test_transform_statement(code, link=True, line_nums=True):
    print('----')
    ra = RegisterAllocator()
    la = label_allocator()
    program = []
    for stmt in ast.parse(code).body:
        program.extend(transform_statement(stmt, ra, la))

    program = optimize(program)

    if link:
        program = link_program(program)

    for index, line in enumerate(program):
        if line_nums:
            line = f'{index}. {line}'
        print(line)


test_transform_statement("""
print(2, 6, 7)
print(2, 6, cell1[3])
printflush(message1)
""")
test_transform_statement("""
2 + 2
3 + 3
""")
test_transform_statement("""
a = 6
b = 2.3 + 5.8
cell1[a] = b
a += 1
cell1[a + 3] *= b + 9
""")
test_transform_statement("""
if a > 3:
    print('Yes')
elif a - b:
    print('Maybe')
else:
    print('No')
""")

exit()


# read time cell1 0
# op add time time 1
# jump 4 lessThan time 300
# set time 0
# write time cell1 0
# jump 8 greaterThan time 200
# control configure unloader1 @titanium 0 0 0
# end
# control configure unloader1 @lead 0 0 0


p2 = """
time = cell1[0]
time += 1
if time >= 300:
    time = 0
cell1[0] = time
if time > 200:
    unloader1.control.configure(Material.lead)
else:
    unloader1.control.configure(Material.titanium)
"""


# TODO: allocate register context for each statement automatically
# as in expressions
