import ast
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

from . import mast, utils


_PY = (sys.version_info.major, sys.version_info.minor)


def _get_ast_slice(node):
    if _PY >= (3, 9):
        return node.slice
    else:
        assert isinstance(node.slice, ast.Index)
        return node.slice.value


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
    ast.Is: 'equal',
}

BOOL_OP_MAP = {  # op, shortcut_condition
    ast.And: ('land', 'equal'),
    ast.Or: ('or', 'notEqual'),
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
RESERVED_NAMES = {
    'print',
    'exit',
    'min',
    'max',
    'atan2',
    'dst',
    'noise',
    'abs',
    'log',
    'log10',
    'sin',
    'cos',
    'tan',
    'floor',
    'ceil',
    'sqrt',
    'rand',
    'print',

    'GetLink',
    'Draw',
    'Control',
    'Radar',
    'Sensor',

    'Material',
    'Liquid',
    'Property',
    'Sort',
    'Target',
    'UnitType',
    'BuildingType',
    'Building',
}


def get_type_map(map, item, desc):
    if type(item) not in map:
        raise ValueError(f'Unsupported {desc} {item}')
    return map[type(item)]


@dataclass
class BaseExpressionHandler:
    expr: Any
    trec: Callable = field(repr=False)
    scope: utils.Scope = field(repr=False)
    pre: list = field(default_factory=list)
    resmap: dict = field(default_factory=dict)

    # AST_CLASS = ast.Xxx

    def dev_dump(self):
        print(ast.dump(self.expr))

    def get_results(self):
        return [self.resmap[i] for i in range(len(self.resmap))]

    def run_trec(self, expr):
        retvals, pre = self.trec(expr, self.scope)
        self.pre.extend(pre)
        return retvals

    def run_trec_single(self, expr):
        retvals = self.run_trec(expr)
        if len(retvals) < 1:
            return mast.Literal(None)
        return retvals[0]

    def proc(self, name, *args):
        self.pre.append(mast.ProcedureCall(name, args))

    def jump(self, label, op, *args):
        self.pre.append(mast.Jump(label, op, args))

    @contextmanager
    def sub_scope(self):
        self.scope = utils.Scope(self.scope)
        try:
            yield
        finally:
            self.scope = self.scope._parent_scope

    def handle(self):
        raise NotImplementedError


class ConstantHandler(BaseExpressionHandler):
    AST_CLASS = ast.Constant

    def handle(self):
        self.resmap[0] = mast.Literal(self.expr.value)


class NameHandler(BaseExpressionHandler):
    AST_CLASS = ast.Name

    def handle(self):
        if self.expr.id in self.scope:
            self.resmap[0] = self.scope[self.expr.id]
            return
        if self.expr.id in RESERVED_NAMES:
            raise ValueError(f'The name {self.expr.id} is reserved')
        self.resmap[0] = mast.Name(self.expr.id)


class TupleHandler(BaseExpressionHandler):
    AST_CLASS = ast.Tuple

    def handle(self):
        for index, value in enumerate(self.expr.elts):
            self.resmap[index] = self.run_trec_single(value)


class SubscriptHandler(BaseExpressionHandler):
    AST_CLASS = ast.Subscript

    def handle(self):
        # memory cell access
        array_val = self.run_trec_single(self.expr.value)
        index_val = self.run_trec_single(_get_ast_slice(self.expr))
        self.resmap[0] = mast.Name()
        self.proc('read', self.resmap[0], array_val, index_val)


class UnaryOpHandler(BaseExpressionHandler):
    AST_CLASS = ast.UnaryOp

    def handle(self):
        factory = get_type_map(UNARY_OP_MAP, self.expr.op, 'UnaryOp')
        val = self.run_trec_single(self.expr.operand)
        self.resmap[0] = mast.Name()
        self.pre.append(factory(val, self.resmap[0]))


class BinOpHandler(BaseExpressionHandler):
    AST_CLASS = ast.BinOp

    def handle(self):
        op = get_type_map(BIN_OP_MAP, self.expr.op, 'BinOp')
        left_val = self.run_trec_single(self.expr.left)
        right_val = self.run_trec_single(self.expr.right)
        self.resmap[0] = mast.Name()
        self.proc(f'op {op}', self.resmap[0], left_val, right_val)


class CompareHandler(BaseExpressionHandler):
    AST_CLASS = ast.Compare

    def handle(self):
        end_label = mast.Label()
        self.resmap[0] = mast.Name()
        a_val = self.run_trec_single(self.expr.left)

        for op, comparator in zip(self.expr.ops, self.expr.comparators):
            op = get_type_map(COND_OP_MAP, op, 'Compare')
            b_val = self.run_trec_single(comparator)
            self.proc(f'op {op}', self.resmap[0], a_val, b_val)
            self.jump(end_label, 'equal', self.resmap[0], mast.Literal(False))

            a_val = b_val

        self.pre.append(end_label)


class BoolOpHandler(BaseExpressionHandler):
    AST_CLASS = ast.BoolOp

    def handle(self):
        op, shortcut_condition = get_type_map(
            BOOL_OP_MAP, self.expr.op, 'BoolOp')

        end_label = mast.Label()
        self.resmap[0] = mast.Name()
        val = self.run_trec_single(self.expr.values[0])
        self.proc('set', self.resmap[0], val)
        self.jump(
            end_label, shortcut_condition, val, mast.Literal(False))

        bool_value = mast.Name()

        for value in self.expr.values[1:]:
            val = self.run_trec_single(value)
            self.proc(f'op {op}', bool_value, self.resmap[0], val)
            self.proc('set', self.resmap[0], val)
            self.jump(
                end_label, shortcut_condition, bool_value, mast.Literal(False))

        self.pre.append(end_label)


class IfExpHandler(BaseExpressionHandler):
    AST_CLASS = ast.IfExp

    def handle(self):
        else_label = mast.Label()
        end_label = mast.Label()

        self.resmap[0] = mast.Name()
        cond = self.run_trec_single(self.expr.test)
        self.jump(else_label, 'equal', cond, mast.Literal(False))

        val = self.run_trec_single(self.expr.body)
        self.proc('set', self.resmap[0], val)
        self.jump(end_label, 'always')

        self.pre.append(else_label)

        val = self.run_trec_single(self.expr.orelse)
        self.proc('set', self.resmap[0], val)

        self.pre.append(end_label)


def _create_unary_op(token):
    def fn(self, a):
        self.resmap[0] = mast.Name()
        self.proc(f'op {token}', self.resmap[0], a)
    return fn


def _create_bin_op(token):
    def fn(self, a, b):
        self.resmap[0] = mast.Name()
        self.proc(f'op {token}', self.resmap[0], a, b)
    return fn


_ZERO = mast.Literal(0)


def build_attr_index(method_map):
    patterns = {}
    for name, method in method_map.items():
        nm = tuple(1 if n == '1' else n for n in name.split('__'))
        cur = patterns
        for n in nm:
            if n not in cur:
                cur[n] = {}
            cur = cur[n]
        assert 'method' not in cur
        cur['method'] = method

    def resolve(nm):
        pnames = []
        cur = patterns
        for n in nm:
            if n.name is not None and n.name in cur:
                cur = cur[n.name]
            elif 1 in cur:
                pnames.append(n)
                cur = cur[1]
            else:
                raise IndexError(f'Unresolvable name {".".join(nm)}')
        if 'method' not in cur:
            raise IndexError(f'Unresolvable name {".".join(nm)}')
        return cur['method'], pnames

    return resolve


def no_at_const(n):
    return mast.Name(n.name.lstrip('@'))


def sort_dir_fn(n):
    if not isinstance(n, mast.Name):
        return n
    elif n.name in ('@asc', 'asc'):
        return mast.Literal(1)
    elif n.name in ('@desc', 'desc'):
        return mast.Literal(-1)
    else:
        return n


class CallHandler(BaseExpressionHandler):
    AST_CLASS = ast.Call

    # TODO: support multiple values
    func__M__min = _create_bin_op('min')
    # TODO: support multiple values
    func__M__max = _create_bin_op('max')
    func__M__atan2 = _create_bin_op('atan2')
    func__M__dst = _create_bin_op('dst')
    func__M__noise = _create_bin_op('noise')
    func__M__abs = _create_unary_op('abs')
    func__M__log = _create_unary_op('log')
    func__M__log10 = _create_unary_op('log10')
    func__M__sin = _create_unary_op('sin')
    func__M__cos = _create_unary_op('cos')
    func__M__tan = _create_unary_op('tan')
    func__M__floor = _create_unary_op('floor')
    func__M__ceil = _create_unary_op('ceil')
    func__M__sqrt = _create_unary_op('sqrt')
    func__M__rand = _create_unary_op('rand')

    def func__M__print(self, *args):
        for arg in args:
            self.proc('print', arg)

    def func__1__printFlush(self, target):
        self.proc('printflush', target)

    def func__M__exit(self):
        self.proc('end')

    def func__M__linkCount(self):
        self.resmap[0] = mast.Name()
        self.proc('set', self.resmap[0], mast.Name('@links'))

    def func__M__getLink(self, index):
        self.resmap[0] = mast.Name()
        self.proc('getlink', self.resmap[0], index)

    def func__1__radar(
            self, unit, target1, target2, target3, sort_type, sort_dir):
        self.resmap[0] = mast.Name()
        self.proc(
            'radar',
            no_at_const(target1), no_at_const(target2), no_at_const(target3),
            no_at_const(sort_type), unit,
            sort_dir_fn(sort_dir), self.resmap[0])

    def func__1__sensor(self, unit, prop):
        self.resmap[0] = mast.Name()
        self.proc('sensor', self.resmap[0], unit, prop)

    def func__M__unit__bind(self, utype):
        self.proc('ubind', utype)

    def func__M__unit__radar(
            self, target1, target2, target3, sort_type, sort_dir):
        self.resmap[0] = mast.Name()
        self.proc(
            'uradar',
            no_at_const(target1), no_at_const(target2), no_at_const(target3),
            no_at_const(sort_type), mast.Name('turret1'),
            sort_dir_fn(sort_dir), self.resmap[0])

    def func__M__locate__building(self, block_type, enemy):
        found = self.resmap[0] = mast.Name()
        x = self.resmap[1] = mast.Name()
        y = self.resmap[2] = mast.Name()
        building = self.resmap[3] = mast.Name()
        self.proc(
            'ulocate building', no_at_const(block_type), enemy,
            mast.Name('@copper'),
            x, y, found, building)

    def func__M__locate__ore(self, material):
        found = self.resmap[0] = mast.Name()
        x = self.resmap[1] = mast.Name()
        y = self.resmap[2] = mast.Name()
        self.proc(
            'ulocate ore', mast.Name('core'), mast.Literal(True),
            material, x, y, found, mast.Name())

    def func__M__locate__spawn(self):
        found = self.resmap[0] = mast.Name()
        x = self.resmap[1] = mast.Name()
        y = self.resmap[2] = mast.Name()
        building = self.resmap[3] = mast.Name()
        self.proc(
            'ulocate spawn', mast.Name('core'), mast.Literal(True),
            mast.Name('@copper'), x, y, found, building)

    def func__M__locate__damaged(self):
        found = self.resmap[0] = mast.Name()
        x = self.resmap[1] = mast.Name()
        y = self.resmap[2] = mast.Name()
        building = self.resmap[3] = mast.Name()
        self.proc(
            'ulocate damaged', mast.Name('core'), mast.Literal(True),
            mast.Name('@copper'), x, y, found, building)

    def func__M__draw__clear(self, r, g, b):
        self.proc('draw clear', r, g, b)

    def func__M__draw__color(self, r, g, b, a):
        self.proc('draw color', r, g, b, a)

    def func__M__draw__stroke(self, width):
        self.proc('draw stroke', width)

    def func__M__draw__line(self, x, y, x2, y2):
        self.proc('draw line', x, y, x2, y2)

    def func__M__draw__rect(self, x, y, width, height):
        self.proc('draw rect', x, y, width, height)

    def func__M__draw__lineRect(self, x, y, width, height):
        self.proc('draw lineRect', x, y, width, height)

    def func__M__draw__poly(self, x, y, sides, radius, rotation):
        self.proc('draw poly', x, y, sides, radius, rotation)

    def func__M__draw__linePoly(self, x, y, sides, radius, rotation):
        self.proc('draw linePoly', x, y, sides, radius, rotation)

    def func__M__draw__triangle(self, x, y, x2, y2, x3, y3):
        self.proc('draw triangle', x, y, x2, y2, x3, y3)

    def func__M__draw__image(self, x, y, image, size, rotation):
        self.proc('draw image', x, y, image, size, rotation)

    def func__1__drawFlush(self, target):
        self.proc('drawflush', target)

    def func__1__setEnabled(self, unit, is_enabled):
        self.proc('control enabled', unit, is_enabled)

    def func__1__targetPosition(self, unit, x, y, shoot):
        self.proc('control shoot', unit, x, y, shoot)

    def func__1__targetObject(self, unit, target, shoot):
        self.proc('control shootp', unit, target, shoot)

    def func__1__configure(self, unit, value):
        self.proc('control configure', unit, value)

    def func__M__unit__stop(self):
        self.proc('ucontrol stop', _ZERO, _ZERO, _ZERO, _ZERO, _ZERO)

    def func__M__unit__move(self, x, y):
        self.proc('ucontrol move', x, y, _ZERO, _ZERO, _ZERO)

    def func__M__unit__approach(self, x, y, radius):
        self.proc('ucontrol approach', x, y, radius, _ZERO, _ZERO)

    def func__M__unit__boost(self, value):
        self.proc('ucontrol boost', value, _ZERO, _ZERO, _ZERO, _ZERO)

    def func__M__unit__pathfind(self):
        self.proc('ucontrol pathfind', _ZERO, _ZERO, _ZERO, _ZERO, _ZERO)

    def func__M__unit__targetPosition(self, x, y, shoot):
        self.proc('ucontrol target', x, y, shoot, _ZERO, _ZERO)

    def func__M__unit__targetObject(self, unit, shoot):
        self.proc('ucontrol targetp', unit, shoot, _ZERO, _ZERO, _ZERO)

    def func__M__unit__itemDrop(self, target, amount):
        self.proc('ucontrol itemDrop', target, amount, _ZERO, _ZERO, _ZERO)

    def func__M__unit__itemTake(self, target, material, amount):
        self.proc('ucontrol itemTake', target, material, amount, _ZERO, _ZERO)

    def func__M__unit__payDrop(self):
        self.proc('ucontrol payDrop', _ZERO, _ZERO, _ZERO, _ZERO, _ZERO)

    def func__M__unit__payTake(self, amount):
        self.proc('ucontrol payTake', amount, _ZERO, _ZERO, _ZERO, _ZERO)

    def func__M__unit__mine(self, x, y):
        self.proc('ucontrol mine', x, y, _ZERO, _ZERO, _ZERO)

    def func__M__unit__setFlag(self, value):
        self.proc('ucontrol flag', value, _ZERO, _ZERO, _ZERO, _ZERO)

    def func__M__unit__build(self, x, y, block, rotation, config):
        self.proc('ucontrol build', x, y, block, rotation, config)

    def func__M__unit__getBlock(self, x, y):
        btype = self.resmap[0] = mast.Name()
        unit = self.resmap[1] = mast.Name()
        self.proc('ucontrol getBlock', x, y, btype, unit, _ZERO)

    def func__M__unit__within(self, x, y, radius):
        self.resmap[0] = mast.Name()
        self.proc('ucontrol within', x, y, radius, self.resmap[0], _ZERO)

    def func__1(self, fname, *args):
        fname = fname.name
        if fname not in self.scope:
            raise NameError(f'Undefined function {fname}')
        fdef = self.scope[fname]
        assert isinstance(fdef, utils.FuncDef)
        if len(args) < fdef.n_args:
            raise TypeError(f'Insufficient arguments for function {fname}')

        for fa, sa in zip(fdef.args, args):
            self.proc('set', fa, sa)

        self.proc(
            'op add', fdef.return_addr, mast.Name('@counter'), mast.Literal(1))
        self.jump(fdef.start_label, 'always')
        for k, v in fdef.resmap.items():
            self.resmap[k] = v

    _resolver = staticmethod(build_attr_index({
        k[len('func__'):]: v
        for k, v in vars().items()
        if k.startswith('func__')
    }))

    def resolve_func(self, value):
        if isinstance(value, ast.Name):
            return [self.run_trec_single(value)]
        elif isinstance(value, ast.Attribute):
            return [*self.resolve_func(value.value), mast.Name(value.attr)]
        else:
            raise ValueError(
                'Expressions resulting in functions are not allowed, '
                'only direct calls of named functions: func(1, 2, 3)'
            )

    def handle(self):
        nm = self.resolve_func(self.expr.func)
        method, pre_args = self._resolver(nm)

        if self.expr.keywords:
            raise ValueError('Keyword arguments are not supported')

        arg_vals = [self.run_trec_single(arg) for arg in self.expr.args]
        method(self, *pre_args, *arg_vals)


class AttributeHandler(BaseExpressionHandler):
    AST_CLASS = ast.Attribute

    def prop__M__at__1(self, attr):
        self.resmap[0] = mast.Name(f'@{attr.replace("_", "-")}')

    def prop__M__n__1(self, attr):
        self.resmap[0] = mast.Name(attr)

    def prop__M__sort__1(self, attr):
        if attr == 'asc':
            self.resmap[0] = mast.Literal(1)
        elif attr == 'desc':
            self.resmap[0] = mast.Literal(-1)
        else:
            self.resmap[0] = mast.Name(attr)

    def prop__1__1(self, unit, prop):
        self.resmap[0] = mast.Name()
        self.proc(
            'sensor', self.resmap[0],
            mast.Name(unit), mast.Name(f'@{prop.replace("_", "-")}'))

    def prop__M__unit__1(self, prop):
        self.prop__1__1('@unit', prop)

    def prop__M__at__unit__1(self, prop):
        self.prop__M__unit__1(prop)

    _resolver = staticmethod(build_attr_index({
        k[len('prop__'):]: v
        for k, v in vars().items()
        if k.startswith('prop__')
    }))

    def resolve_value(self, value):
        if isinstance(value, ast.Name):
            return [self.run_trec_single(value)]
        elif isinstance(value, ast.Attribute):
            return [*self.resolve_value(value.value), mast.Name(value.attr)]
        else:
            raise ValueError(
                'Expressions are not allowed before attribute access, '
                'use names of objects directly: Material.copper'
            )

    def handle(self):
        nm = self.resolve_value(self.expr)
        method, pre_args = self._resolver(nm)
        method(self, *(nm.name for nm in pre_args))


# Statements ===================================


class ExprStatementHandler(BaseExpressionHandler):
    AST_CLASS = ast.Expr

    def handle(self):
        self.run_trec(self.expr.value)


def _check_assignment_to_reserved(name):
    if name in RESERVED_NAMES:
        raise ValueError(f'The name {name} is reserved')


class AssignStatementHandler(BaseExpressionHandler):
    AST_CLASS = ast.Assign

    def _assign(self, targets, values):
        assert len(targets) <= len(values)
        for target, value in zip(targets, values):
            target = self.run_trec_single(target)
            self.proc('set', target, value)

    def named_assign(self, target, value):
        _check_assignment_to_reserved(target.id)
        retvals = self.run_trec(value)
        self._assign([target], retvals)

    def tuple_assign(self, target, value):
        assert all(isinstance(n, ast.Name) for n in target.elts)
        retvals = self.run_trec(value)
        self._assign(target.elts, retvals)

    def memory_assign(self, target, value):
        if not isinstance(target.value, ast.Name):
            raise ValueError(f'Unsupported assignment target {target}')
        _check_assignment_to_reserved(target.value.id)
        index_val = self.run_trec_single(_get_ast_slice(target))
        value_val = self.run_trec_single(value)
        self.proc('write', value_val, mast.Name(target.value.id), index_val)

    TARGET_MAP = {
        ast.Name: named_assign,
        ast.Subscript: memory_assign,
        ast.Tuple: tuple_assign,
    }

    def handle(self):
        if len(self.expr.targets) != 1:
            raise ValueError(
                'Only single target can be used in assignment: a = 3')
        target = self.expr.targets[0]
        if type(target) not in self.TARGET_MAP:
            raise ValueError(f'Unsupported assignment target {target}')
        method = self.TARGET_MAP[type(target)]
        method(self, target, self.expr.value)


class AugAssignStatementHandler(BaseExpressionHandler):
    AST_CLASS = ast.AugAssign

    def named_assign(self, target, op, operand):
        operand_val = self.run_trec_single(operand)
        t = self.run_trec_single(target)
        self.proc(f'op {op}', t, t, operand_val)

    def memory_assign(self, target, op, operand):
        if not isinstance(target.value, ast.Name):
            raise ValueError(f'Unsupported assignment target {target}')
        index_val = self.run_trec_single(_get_ast_slice(target))
        operand_val = self.run_trec_single(operand)
        op_output = mast.Name()
        cell = self.run_trec_single(target.value)
        self.proc('read', op_output, cell, index_val)
        self.proc(f'op {op}', op_output, op_output, operand_val)
        self.proc('write', op_output, cell, index_val)

    TARGET_MAP = {
        ast.Name: named_assign,
        ast.Subscript: memory_assign,
    }

    def handle(self):
        target = self.expr.target
        if type(target) not in self.TARGET_MAP:
            raise ValueError(f'Unsupported assignment target {target}')
        method = self.TARGET_MAP[type(target)]
        op = get_type_map(BIN_OP_MAP, self.expr.op, 'BinOp')
        method(self, target, op, self.expr.value)


class IfStatementHandler(BaseExpressionHandler):
    AST_CLASS = ast.If

    def handle(self):
        end_label = mast.Label()
        else_label = mast.Label() if self.expr.orelse else end_label

        test_val = self.run_trec_single(self.expr.test)
        self.jump(else_label, 'equal', test_val, mast.Literal(False))

        for stmt in self.expr.body:
            self.run_trec(stmt)

        if self.expr.orelse:
            self.jump(end_label, 'always')
            self.pre.append(else_label)

            for stmt in self.expr.orelse:
                self.run_trec(stmt)

        self.pre.append(end_label)


class WhileStatementHandler(BaseExpressionHandler):
    AST_CLASS = ast.While

    def handle(self):
        loop_label = mast.Label()
        end_label = mast.Label()

        self.pre.append(loop_label)
        test_val = self.run_trec_single(self.expr.test)
        self.jump(end_label, 'equal', test_val, mast.Literal(False))

        for stmt in self.expr.body:
            self.run_trec(stmt)

        self.jump(loop_label, 'always')
        self.pre.append(end_label)


class FunctionDefStatementHandler(BaseExpressionHandler):
    AST_CLASS = ast.FunctionDef

    def handle(self):
        if _PY >= (3, 8):
            assert self.expr.args.posonlyargs == []
            assert self.expr.type_comment is None
        assert self.expr.args.vararg is None
        assert self.expr.args.kwonlyargs == []
        assert self.expr.args.kw_defaults == []
        assert self.expr.args.kwarg is None
        assert self.expr.args.defaults == []
        assert self.expr.decorator_list == []
        assert self.expr.returns is None

        fdef = utils.FuncDef(self.expr.name, len(self.expr.args.args))
        self.scope[self.expr.name] = fdef

        end_label = mast.Label()
        self.jump(end_label, 'always')  # skip function body
        self.pre.append(fdef.start_label)

        before, self.pre = self.pre, []

        with self.sub_scope():
            self.scope._current_func = fdef
            for a, n in zip(self.expr.args.args, fdef.args):
                self.scope[a.arg] = n

            for stmt in self.expr.body:
                self.run_trec(stmt)

            fdef.create_return(self)

        self.pre.append(end_label)

        # we have to do this here, since we need to know
        # max size of return tuple
        body, self.pre = self.pre, before
        for rv in fdef.resmap.values():
            self.proc('set', rv, mast.Literal(None))
        self.pre.extend(body)


class ReturnHandler(BaseExpressionHandler):
    AST_CLASS = ast.Return

    def handle(self):
        fdef = self.scope.current_func
        if fdef is None:
            raise ValueError('Returning outside function context')

        for index, value in enumerate(self.run_trec(self.expr.value)):
            self.resmap[index] = value
            if index not in fdef.resmap:
                fdef.resmap[index] = mast.Name()
            self.proc('set', fdef.resmap[index], value)

        fdef.create_return(self)


class ImportFromHandler(BaseExpressionHandler):
    AST_CLASS = ast.ImportFrom

    def handle(self):
        pass  # intentionally do nothing


AST_NODE_MAP = {
    subcls.AST_CLASS: subcls
    for subcls in BaseExpressionHandler.__subclasses__()
}


def _create_ast_constant_hack(ast_cls, conv):
    def cons(anode, *args, **kwargs):
        return ConstantHandler(ast.Constant(conv(anode)), *args, **kwargs)
    AST_NODE_MAP[ast_cls] = cons


if _PY < (3, 8):
    _create_ast_constant_hack(ast.Num, lambda expr: expr.n)
    _create_ast_constant_hack(ast.NameConstant, lambda expr: expr.value)
    _create_ast_constant_hack(ast.Str, lambda expr: expr.s)
    _create_ast_constant_hack(ast.Ellipsis, lambda expr: ...)


def transform_expr(expr, scope):
    htype = get_type_map(AST_NODE_MAP, expr, 'expression')
    node = htype(expr, transform_expr, scope)
    node.handle()
    return node.get_results(), node.pre
