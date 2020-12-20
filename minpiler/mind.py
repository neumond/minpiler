import ast
from dataclasses import dataclass
from typing import Any, Callable

from . import mast


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
    _result: str = None

    # AST_CLASS = ast.Xxx

    def dev_dump(self):
        print(ast.dump(self.expr))

    @property
    def result(self):
        if self._result is None:
            self._result = mast.Name()
        return self._result

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
        return self.result, [
            *array_pre,
            *index_pre,
            mast.FunctionCall(
                'read', [array_val, index_val], self.result),
        ]


class UnaryOpHandler(BaseExpressionHandler):
    AST_CLASS = ast.UnaryOp

    def handle(self):
        factory = get_type_map(UNARY_OP_MAP, self.expr.op, 'UnaryOp')
        val, pre = self.trec(self.expr.operand)
        return self.result, [
            *pre,
            factory(val, self.result),
        ]


class BinOpHandler(BaseExpressionHandler):
    AST_CLASS = ast.BinOp

    def handle(self):
        op = get_type_map(BIN_OP_MAP, self.expr.op, 'BinOp')
        left_val, left_pre = self.trec(self.expr.left)
        right_val, right_pre = self.trec(self.expr.right)
        return self.result, [
            *left_pre,
            *right_pre,
            mast.FunctionCall(
                f'op {op}', [left_val, right_val], self.result),
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
                f'op {op}', [a_val, b_val], self.result))
            pre.append(mast.Jump(
                end_label, 'equal', [self.result, mast.Literal(False)]))
            a_val = b_val

        pre.append(end_label)
        return self.result, pre


class BoolOpHandler(BaseExpressionHandler):
    AST_CLASS = ast.BoolOp

    def handle(self):
        op, shortcut_condition = get_type_map(
            BOOL_OP_MAP, self.expr.op, 'BoolOp')

        end_label = mast.Label()
        val, pre = self.trec(self.expr.values[0])
        pre.append(mast.FunctionCall('set', [val], self.result))

        bool_value = mast.Name()

        for value in self.expr.values[1:]:
            val, b_pre = self.trec(value)
            pre.extend(b_pre)
            pre.append(mast.FunctionCall(
                f'op {op}', [self.result, val], bool_value))
            pre.append(mast.Jump(
                end_label, 'equal', [bool_value, shortcut_condition]))
            pre.append(mast.FunctionCall(
                'set', [val], self.result))

        pre.append(end_label)
        return self.result, pre


class CallHandler(BaseExpressionHandler):
    AST_CLASS = ast.Call

    def func_min(self, a, b):
        # TODO: support multiple values
        return [mast.FunctionCall('op min', [a, b], self.result)]

    def func_max(self, a, b):
        # TODO: support multiple values
        return [mast.FunctionCall('op max', [a, b], self.result)]

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

        if self._result is None:
            return mast.Literal(None), result_pre
        else:
            return self.result, result_pre


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


def transform_expr(expr):
    def trec(expr):
        return transform_expr(expr)

    if type(expr) not in AST_NODE_MAP:
        raise ValueError(f'Unsupported expression {expr}')

    return AST_NODE_MAP[type(expr)](expr, trec).handle()


def test_transform_expr(code):
    assert len(ast.parse(code).body) == 1
    expr = ast.parse(code).body[0]
    assert isinstance(expr, ast.Expr)
    val, lines = transform_expr(expr.value)
    print('-----')
    lines.append(mast.ProcedureCall('print', [val]))
    for line in mast.dump(lines):
        print(line)


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
# test_transform_expr('True and True or False and 3')
# exit()


@dataclass
class BaseStatementHandler:
    stmt: Any

    # AST_CLASS = ast.Xxx

    def dev_dump(self):
        print(ast.dump(self.stmt))

    def handle(self):
        raise NotImplementedError


class ExprStatementHandler(BaseStatementHandler):
    AST_CLASS = ast.Expr

    def handle(self):
        retval, pre = transform_expr(self.stmt.value)
        return pre


class AssignStatementHandler(BaseStatementHandler):
    AST_CLASS = ast.Assign

    def named_assign(self, target, value):
        retval, pre = transform_expr(value)
        pre.append(mast.FunctionCall('set', [retval], mast.Name(target.id)))
        return pre

    def memory_assign(self, target, value):
        if not isinstance(target.value, ast.Name):
            raise ValueError(f'Unsupported assignment target {target}')
        assert isinstance(target.slice, ast.Index)
        index_val, index_pre = transform_expr(target.slice.value)
        value_val, value_pre = transform_expr(value)
        return [
            *index_pre,
            *value_pre,
            mast.ProcedureCall('write', [
                value_val, mast.Name(target.value.id), index_val]),
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
        operand_val, pre = transform_expr(operand)
        t = mast.Name(target.id)
        pre.append(mast.FunctionCall(f'op {op}', [t, operand_val], t))
        return pre

    def memory_assign(self, target, op, operand):
        if not isinstance(target.value, ast.Name):
            raise ValueError(f'Unsupported assignment target {target}')
        assert isinstance(target.slice, ast.Index)
        index_val, index_pre = transform_expr(target.slice.value)
        operand_val, operand_pre = transform_expr(operand)
        op_output = mast.Name()
        cell = mast.Name(target.value.id)
        return [
            *index_pre,
            *operand_pre,
            mast.FunctionCall('read', [cell, index_val], op_output),
            mast.FunctionCall(f'op {op}', [op_output, operand_val], op_output),
            mast.ProcedureCall('write', [op_output, cell, index_val]),
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
        end_label = mast.Label()
        else_label = mast.Label() if self.stmt.orelse else end_label

        result = []

        test_val, test_pre = transform_expr(self.stmt.test)
        result.extend(test_pre)
        result.append(mast.Jump(
            else_label, 'equal', [test_val, mast.Literal(False)]))

        for stmt in self.stmt.body:
            result.extend(transform_statement(stmt))

        if self.stmt.orelse:
            result.append(mast.Jump(end_label, 'always', []))
            result.append(else_label)

            for stmt in self.stmt.orelse:
                result.extend(transform_statement(stmt))

        result.append(end_label)
        return result


AST_STATEMENT_MAP = {
    subcls.AST_CLASS: subcls
    for subcls in BaseStatementHandler.__subclasses__()
}


def transform_statement(stmt):
    if type(stmt) not in AST_STATEMENT_MAP:
        raise ValueError(f'Unsupported statement {stmt}')

    return AST_STATEMENT_MAP[type(stmt)](stmt).handle()


def test_transform_statement(code, line_nums=True):
    print('----')
    program = []
    for stmt in ast.parse(code).body:
        program.extend(transform_statement(stmt))
    for index, line in enumerate(mast.dump(program)):
        if line_nums:
            line = f'{index}. {line}'
        print(line)


# test_transform_statement("""
# print(2, 6, 7)
# print(2, 6, cell1[3])
# printflush(message1)
# """)
# test_transform_statement("""
# 2 + 2
# 3 + 3
# """)
# test_transform_statement("""
# a = 6
# b = 2.3 + 5.8
# cell1[a] = b
# a += 1
# cell1[a + 3] *= b + 9
# """)
# test_transform_statement("""
# if a > 3:
#     print('Yes')
# elif a - b:
#     print('Maybe')
# else:
#     print('No')
# """)
# exit()


# draw clear 0 0 0 0 0 0
# draw color 255 255 255 255 0 0
# draw image 100 100 @copper 32 0 0
# draw image 50 50 @lead 32 0 0
# draw color 0 255 0 100 0 0
# draw rect 0 0 10 10 0 0
# drawflush display1
# radar any any any distance duo1 1 result
# print result
# printflush message1
# op equal result a b


# p2 = """
# time = cell1[0]
# time += 1
# if time >= 300:
#     time = 0
# cell1[0] = time
# if time > 200:
#     unloader1.control.configure(Material.lead)
# else:
#     unloader1.control.configure(Material.titanium)
# """
