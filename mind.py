import ast
import string
from dataclasses import dataclass
from itertools import count, islice
from typing import Any, Callable


def label_allocator():
    for i in count(start=1):
        yield f'<label:{i}>'


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


OPERATOR_MAP = {
    ast.Gt: 'greaterThan',
    ast.GtE: 'greaterThanEq',
    ast.Lt: 'lessThan',
    ast.LtE: 'lessThanEq',
    ast.Eq: 'equal',
    ast.NotEq: 'notEqual',
}

NEGATE_MAP = {
    'greaterThan': 'lessThanEq',
    'greaterThanEq': 'lessThan',
    'lessThan': 'greaterThanEq',
    'lessThanEq': 'greaterThan',
    'equal': 'notEqual',
    'notEqual': 'equal',
}


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
            raise ValueError(f'Quotes are not allowed in literal strings {c!r}')
        return '"' + c.replace('\n', r'\n') + '"'
    else:
        raise ValueError(f'Unsupported constant {c!r}')


def transform_name_or_constant(val):
    if isinstance(val, ast.Name):
        return val.id
    elif isinstance(val, ast.Constant):
        return transform_constant(val)
    else:
        raise ValueError(f'Not a name nor a constant {val}')


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

BOOL_OP_MAP = {
    ast.And: 'land',
    ast.Or: 'or',
}

UNARY_OP_MAP = {
    ast.Invert: ('not', '{}'),
    ast.Not: ('equal', '0 {}'),
    ast.UAdd: ('add', '0 {}'),
    ast.USub: ('sub', '0 {}'),
}


@dataclass
class AstNodeHandler:  # TODO: rename to AstExprHandler
    expr: Any
    trec: Callable
    alloc_result_name: Callable
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


class ConstantHandler(AstNodeHandler):
    AST_CLASS = ast.Constant

    def handle(self):
        return transform_constant(self.expr), []


class NameHandler(AstNodeHandler):
    AST_CLASS = ast.Name

    def handle(self):
        return self.expr.id, []


class SubscriptHandler(AstNodeHandler):
    AST_CLASS = ast.Subscript

    def handle(self):
        # memory cell access
        assert isinstance(self.expr.slice, ast.Index)
        array_val, array_pre = self.trec(self.expr.value)
        index_val, index_pre = self.trec(self.expr.slice.value)
        return self.result_name, [
            *array_pre,
            *index_pre,
            f'read {self.result_name} {array_val} {index_val}',
        ]


class UnaryOpHandler(AstNodeHandler):
    AST_CLASS = ast.UnaryOp

    def handle(self):
        if type(self.expr.op) not in UNARY_OP_MAP:
            raise ValueError(f'Unsupported UnaryOp {self.expr.op}')
        val, pre = self.trec(self.expr.operand)
        op, template = UNARY_OP_MAP[type(self.expr.op)]
        return self.result_name, [
            *pre,
            f'op {op} {self.result_name} {template.format(val)}',
        ]


class BinOpHandler(AstNodeHandler):
    AST_CLASS = ast.BinOp

    def handle(self):
        if type(self.expr.op) not in BIN_OP_MAP:
            raise ValueError(f'Unsupported BinOp {self.expr.op}')
        left_val, left_pre = self.trec(self.expr.left)
        right_val, right_pre = self.trec(self.expr.right)
        op = BIN_OP_MAP[type(self.expr.op)]
        return self.result_name, [
            *left_pre,
            *right_pre,
            f'op {op} {self.result_name} {left_val} {right_val}',
        ]


class CallHandler(AstNodeHandler):
    AST_CLASS = ast.Call

    def func_min(self, a, b):
        # TODO: support multiple values
        return [f'op min {self.result_name} {a} {b}']

    def func_max(self, a, b):
        # TODO: support multiple values
        return [f'op max {self.result_name} {a} {b}']

    def func_print(self, *args):
        return [f'print {arg}' for arg in args]

    def func_printflush(self, target):
        return [f'printflush {target}']

    def func_exit(self):
        return ['end']

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
            return 'null', result_pre
        else:
            return self.result_name, result_pre


class AttributeHandler(AstNodeHandler):
    AST_CLASS = ast.Attribute

    def obj_Material(self, attr):
        return f'@{attr}', []

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
    for subcls in AstNodeHandler.__subclasses__()
}


def transform_expr(expr, register_allocator, alloc_result_name):
    with register_allocator.context() as reg:

        def trec(expr):
            return transform_expr(expr, register_allocator, reg.allocate)

        if type(expr) not in AST_NODE_MAP:
            raise ValueError(f'Unsupported expression {expr}')

        return AST_NODE_MAP[type(expr)](expr, trec, alloc_result_name).handle()


def test_transform_expr(code):
    assert len(ast.parse(code).body) == 1
    expr = ast.parse(code).body[0]
    assert isinstance(expr, ast.Expr)
    val, lines = transform_expr(
        expr.value,
        RegisterAllocator(),
        lambda: 'result',
    )
    for line in lines:
        print(line)
    print('print', val)


# test_transform_expr('2+2')
# test_transform_expr('2 + 2 * 2 + 8 + 6 * 9 * 3')
# test_transform_expr('-5')
# test_transform_expr('cell["kek"]')
# test_transform_expr('max(min(2, 8), 3 + 3)')
# test_transform_expr('print(1, 2 + 7, 3, print(), "lol")')
# test_transform_expr('printflush(message1)')
# test_transform_expr('Material.copper')
# test_transform_expr('exit()')

# TODO:
# test_transform_expr('True and True')

# exit()


@dataclass
class BaseStatementHandler:
    stmt: Any
    eval_expr: Callable

    # AST_CLASS = ast.Xxx

    def dev_dump(self):
        print(ast.dump(self.stmt))

    def handle(self):
        raise NotImplementedError


class ExprStatementHandler(BaseStatementHandler):
    AST_CLASS = ast.Expr

    def handle(self):
        retval, pre = self.eval_expr(self.stmt.value)
        return pre


class AssignStatementHandler(BaseStatementHandler):
    AST_CLASS = ast.Assign

    def named_assign(self, target, value):
        retval, pre = self.eval_expr(value, target.id)
        if retval != target.id:
            pre.append(f'set {target.id} {retval}')
        return pre

    def memory_assign(self, target, value):
        self.dev_dump()
        if not isinstance(target.value, ast.Name):
            raise ValueError(f'Unsupported assignment target {target}')
        assert isinstance(target.slice, ast.Index)
        index_val, index_pre = self.eval_expr(target.slice.value)
        value_val, value_pre = self.eval_expr(value)
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


AST_STATEMENT_MAP = {
    subcls.AST_CLASS: subcls
    for subcls in BaseStatementHandler.__subclasses__()
}


def transform_statement(stmt, register_allocator):
    if type(stmt) not in AST_STATEMENT_MAP:
        raise ValueError(f'Unsupported statement {stmt}')

    def eval_expr(expr, result_name=None):
        if result_name is None:
            with register_allocator.context() as reg:
                return transform_expr(expr, register_allocator, reg.allocate)
        else:
            register_allocator._busy.add(result_name)
            return transform_expr(
                expr, register_allocator, lambda: result_name)

    return AST_STATEMENT_MAP[type(stmt)](stmt, eval_expr).handle()


def test_transform_statement(code):
    print('----')
    ra = RegisterAllocator()
    for stmt in ast.parse(code).body:
        lines = transform_statement(stmt, ra)
        for line in lines:
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
""")

exit()


class Transform:
    def __init__(self, python_code):
        self.python_code = python_code
        self.label_allocator = label_allocator()
        self.register_allocator = RegisterAllocator([
            'null', 'true', 'false',
            # TODO: make busy all things like cell1, conveyor3
        ])

    def transform_assign(self, targets, value):
        assert len(targets) == 1

        if isinstance(targets[0], ast.Subscript):
            # write to memory cell
            assert isinstance(targets[0].slice, ast.Index)
            assert isinstance(value, ast.Name)
            return [
                f'write {value.id} '
                f'{targets[0].value} '
                f'{transform_constant(targets[0].slice.value)}',
            ]
        elif isinstance(targets[0], ast.Name):
            # write to variable
            if isinstance(value, ast.Subscript):
                # memory cell access
                assert isinstance(value.slice, ast.Index)
                return [
                    f'read {targets[0].id} {value.value} '
                    f'{transform_constant(value.slice.value)}',
                ]
            else:
                return [f'set {targets[0].id} {transform_constant(value)}']
        else:
            raise ValueError(f'Unsupported assignment {targets[0]}')

    def transform_aug_assign(self, target, op, value):
        assert isinstance(target, ast.Name)
        if isinstance(op, ast.Add):
            assert isinstance(value, ast.Constant)
            return [f'op add {target.id} {target.id} {value.value}']
        elif isinstance(op, ast.Mult):
            assert isinstance(value, ast.Constant)
            return [f'op mul {target.id} {target.id} {value.value}']
        else:
            raise ValueError(f'Unsupported op {op}')

    def transform_print(self, expr):
        assert expr.func.id == 'print'
        assert expr.keywords == []
        return [
            f'print {transform_name_or_constant(val)}'
            for val in expr.args
        ]

    def transform_printflush(self, expr):
        assert expr.func.id == 'printflush'
        assert expr.keywords == []
        assert len(expr.args) == 1
        assert expr.args[0].__class__ is ast.Name
        return [f'{expr.func.id} {expr.args[0].id}']

    def transform_expr(self, expr):
        cls = expr.__class__
        assert cls is ast.Call
        assert expr.func.__class__ is ast.Name
        if expr.func.id == 'print':
            return self.transform_print(expr)
        elif expr.func.id == 'printflush':
            return self.transform_printflush(expr)
        else:
            raise ValueError(f'Unsupported function {expr.func.id}')

    def transform_if_test(self, test, negate=False):
        assert isinstance(test, ast.Compare)

        assert isinstance(test.left, ast.Name)
        assert len(test.ops) == 1
        assert isinstance(
            test.ops[0], tuple(cls for cls in OPERATOR_MAP.keys()))
        assert len(test.comparators) == 1
        assert isinstance(test.comparators[0], ast.Constant)

        op = OPERATOR_MAP[type(test.ops[0])]
        if negate:
            op = NEGATE_MAP[op]

        return f'{op} {test.left.id} {test.comparators[0].value}'

    def transform_if(self, test, body, orelse):
        # jump 8 greaterThan time 200
        # set x @lead
        # print x
        # printflush message1
        # jump 0 always

        body_statements = [self.transform_stmt(stmt) for stmt in body]
        if not orelse:
            label_end = next(self.label_allocator)
            return [
                f'jump {label_end} '
                + self.transform_if_test(test, negate=True),
                *body_statements,
                label_end,
            ]
        else:
            print(orelse)
            assert False
            # label_main = next(self.label_allocator)
            # label_end = next(self.label_allocator)
            # return [
            #     f'jump {label_main} ' + self.transform_if_test(test),
            #     *body_statements
            # ]

    def transform_stmt(self, stmt):
        if isinstance(stmt, ast.Assign):
            return self.transform_assign(stmt.targets, stmt.value)
        if isinstance(stmt, ast.AugAssign):
            return self.transform_aug_assign(stmt.target, stmt.op, stmt.value)
        if isinstance(stmt, ast.If):
            return self.transform_if(stmt.test, stmt.body, stmt.orelse)
        if isinstance(stmt, ast.Expr):
            return self.transform_expr(stmt.value)
        raise ValueError(f'Unsupported statement {stmt}')

    def __call__(self):
        self.lines = []
        mod = ast.parse(self.python_code)
        for stmt in mod.body:
            self.lines.extend(self.transform_stmt(stmt))
        return '\n'.join(self.lines)


# set x 30
# op sub x x 10
# op add x x 0
# print x
# printflush message1


# read time cell1 0
# op add time time 1
# jump 4 lessThan time 300
# set time 0
# write time cell1 0
# jump 8 greaterThan time 200
# control configure unloader1 @titanium 0 0 0
# end
# control configure unloader1 @lead 0 0 0


p1 = """
x = 30
x += 6
x *= 4
print('lol', x)
printflush(message1)
"""

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

print(Transform(p1)())
