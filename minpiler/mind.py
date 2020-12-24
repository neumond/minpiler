import ast
import sys
from dataclasses import dataclass, field
from typing import Any, Callable

from . import mast


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
    pre: list = field(default_factory=list)
    resmap: dict = field(default_factory=dict)

    # AST_CLASS = ast.Xxx

    def dev_dump(self):
        print(ast.dump(self.expr))

    def get_results(self):
        return [self.resmap[i] for i in range(len(self.resmap))]

    def run_trec(self, expr):
        retvals, pre = self.trec(expr)
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

    def handle(self):
        raise NotImplementedError


class ConstantHandler(BaseExpressionHandler):
    AST_CLASS = ast.Constant

    def handle(self):
        self.resmap[0] = mast.Literal(self.expr.value)


class NameHandler(BaseExpressionHandler):
    AST_CLASS = ast.Name

    def handle(self):
        if self.expr.id in RESERVED_NAMES:
            raise ValueError(f'The name {self.expr.id} is reserved')
        self.resmap[0] = mast.Name(self.expr.id)


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


class CallHandler(BaseExpressionHandler):
    AST_CLASS = ast.Call

    # TODO: support multiple values
    func_min = _create_bin_op('min')
    # TODO: support multiple values
    func_max = _create_bin_op('max')
    func_atan2 = _create_bin_op('atan2')
    func_dst = _create_bin_op('dst')
    func_noise = _create_bin_op('noise')
    func_abs = _create_unary_op('abs')
    func_log = _create_unary_op('log')
    func_log10 = _create_unary_op('log10')
    func_sin = _create_unary_op('sin')
    func_cos = _create_unary_op('cos')
    func_tan = _create_unary_op('tan')
    func_floor = _create_unary_op('floor')
    func_ceil = _create_unary_op('ceil')
    func_sqrt = _create_unary_op('sqrt')
    func_rand = _create_unary_op('rand')

    def func_print(self, *args):
        for arg in args:
            self.proc('print', arg)

    def func_exit(self):
        self.proc('end')

    def func_GetLink(self, index):
        self.resmap[0] = mast.Name()
        self.proc('getlink', self.resmap[0], index)

    def func_Radar(self, unit, target1, target2, target3, sort_type, sort_dir):
        self.resmap[0] = mast.Name()
        self.proc(
            'radar', target1, target2, target3,
            sort_type, unit, sort_dir, self.resmap[0])

    def func_Sensor(self, unit, prop):
        self.resmap[0] = mast.Name()
        self.proc('sensor', self.resmap[0], unit, prop)

    def func_UnitBind(self, utype):
        self.proc('ubind', utype)

    def func_UnitRadar(self, target1, target2, target3, sort_type, sort_dir):
        self.resmap[0] = mast.Name()
        self.proc(
            'uradar', target1, target2, target3,
            sort_type, mast.Name('turret1'), sort_dir, self.resmap[0])

    def func_LocateBuilding(self, block_type, enemy):
        found = self.resmap[0] = mast.Name()
        x = self.resmap[1] = mast.Name()
        y = self.resmap[2] = mast.Name()
        building = self.resmap[3] = mast.Name()
        self.proc(
            'ulocate building', block_type, enemy,
            mast.Name('@copper'),
            x, y, found, building)

    def func_LocateOre(self, material):
        found = self.resmap[0] = mast.Name()
        x = self.resmap[1] = mast.Name()
        y = self.resmap[2] = mast.Name()
        self.proc(
            'ulocate ore', mast.Name('core'), mast.Literal(True),
            material, x, y, found, mast.Name())

    def func_LocateSpawn(self):
        found = self.resmap[0] = mast.Name()
        x = self.resmap[1] = mast.Name()
        y = self.resmap[2] = mast.Name()
        building = self.resmap[3] = mast.Name()
        self.proc(
            'ulocate spawn', mast.Name('core'), mast.Literal(True),
            mast.Name('@copper'), x, y, found, building)

    def func_LocateDamaged(self):
        found = self.resmap[0] = mast.Name()
        x = self.resmap[1] = mast.Name()
        y = self.resmap[2] = mast.Name()
        building = self.resmap[3] = mast.Name()
        self.proc(
            'ulocate damaged', mast.Name('core'), mast.Literal(True),
            mast.Name('@copper'), x, y, found, building)

    def method_print_flush(self, target):
        self.proc('printflush', target)

    def method_Draw_clear(self, r, g, b):
        self.proc('draw clear', r, g, b)

    def method_Draw_color(self, r, g, b, a):
        self.proc('draw color', r, g, b, a)

    def method_Draw_stroke(self, width):
        self.proc('draw stroke', width)

    def method_Draw_line(self, x, y, x2, y2):
        self.proc('draw line', x, y, x2, y2)

    def method_Draw_rect(self, x, y, width, height):
        self.proc('draw rect', x, y, width, height)

    def method_Draw_lineRect(self, x, y, width, height):
        self.proc('draw lineRect', x, y, width, height)

    def method_Draw_poly(self, x, y, sides, radius, rotation):
        self.proc('draw poly', x, y, sides, radius, rotation)

    def method_Draw_linePoly(self, x, y, sides, radius, rotation):
        self.proc('draw linePoly', x, y, sides, radius, rotation)

    def method_Draw_triangle(self, x, y, x2, y2, x3, y3):
        self.proc('draw triangle', x, y, x2, y2, x3, y3)

    def method_Draw_image(self, x, y, image, size, rotation):
        self.proc('draw image', x, y, image, size, rotation)

    def method_Draw_flush(self, target):
        self.proc('drawflush', target)

    def method_Control_setEnabled(self, unit, is_enabled):
        self.proc('control enabled', unit, is_enabled)

    def method_Control_targetPosition(self, unit, x, y, shoot):
        self.proc('control shoot', unit, x, y, shoot)

    def method_Control_targetObject(self, unit, target, shoot):
        self.proc('control shootp', unit, target, shoot)

    def method_Control_configure(self, unit, value):
        self.proc('control configure', unit, value)

    def method_UnitControl_stop(self):
        self.proc('ucontrol stop', _ZERO, _ZERO, _ZERO, _ZERO, _ZERO)

    def method_UnitControl_move(self, x, y):
        self.proc('ucontrol move', x, y, _ZERO, _ZERO, _ZERO)

    def method_UnitControl_approach(self, x, y, radius):
        self.proc('ucontrol approach', x, y, radius, _ZERO, _ZERO)

    def method_UnitControl_boost(self, value):
        self.proc('ucontrol boost', value, _ZERO, _ZERO, _ZERO, _ZERO)

    def method_UnitControl_pathfind(self):
        self.proc('ucontrol pathfind', _ZERO, _ZERO, _ZERO, _ZERO, _ZERO)

    def method_UnitControl_target(self, x, y, shoot):
        self.proc('ucontrol target', x, y, shoot, _ZERO, _ZERO)

    def method_UnitControl_targetp(self, unit, shoot):
        self.proc('ucontrol targetp', unit, shoot, _ZERO, _ZERO, _ZERO)

    def method_UnitControl_itemDrop(self, unit, amount):
        self.proc('ucontrol itemDrop', unit, amount, _ZERO, _ZERO, _ZERO)

    def method_UnitControl_itemTake(self, unit, material, amount):
        self.proc('ucontrol itemTake', unit, material, amount, _ZERO, _ZERO)

    def method_UnitControl_payDrop(self):
        self.proc('ucontrol payDrop', _ZERO, _ZERO, _ZERO, _ZERO, _ZERO)

    def method_UnitControl_payTake(self, amount):
        self.proc('ucontrol payTake', amount, _ZERO, _ZERO, _ZERO, _ZERO)

    def method_UnitControl_mine(self, x, y):
        self.proc('ucontrol mine', x, y, _ZERO, _ZERO, _ZERO)

    def method_UnitControl_flag(self, value):
        self.proc('ucontrol flag', value, _ZERO, _ZERO, _ZERO, _ZERO)

    def method_UnitControl_build(self, x, y, block, rotation, config):
        self.proc('ucontrol build', x, y, block, rotation, config)

    def method_UnitControl_getBlock(self, x, y):
        btype = self.resmap[0] = mast.Name()
        unit = self.resmap[1] = mast.Name()
        self.proc('ucontrol getBlock', x, y, btype, unit, _ZERO)

    def method_UnitControl_within(self, x, y, radius):
        self.resmap[0] = mast.Name()
        self.proc('ucontrol within', x, y, radius, self.resmap[0], _ZERO)

    def _get_object_method(self, expr):
        if not isinstance(expr.value, ast.Name):
            raise ValueError(
                'Expressions are not allowed before attribute access, '
                'use names of objects directly: Draw.clear(0, 0, 0)'
            )
        return f'{expr.value.id}_{expr.attr}'

    def handle(self):
        if isinstance(self.expr.func, ast.Attribute):
            fname = self._get_object_method(self.expr.func)
            method = getattr(self, 'method_' + fname, None)
            if method is None:
                raise ValueError(f'Unknown method name {fname}')
        elif isinstance(self.expr.func, ast.Name):
            fname = self.expr.func.id
            method = getattr(self, 'func_' + fname, None)
            if method is None:
                raise ValueError(f'Unknown function name {fname}')
        else:
            raise ValueError(
                'Expressions resulting in functions are not allowed, '
                'only direct calls of named functions: func(1, 2, 3)'
            )

        if self.expr.keywords:
            raise ValueError('Keyword arguments are not supported')

        arg_vals = [self.run_trec_single(arg) for arg in self.expr.args]
        method(*arg_vals)


_method_values_permitted = 'Using {} methods as values is permitted'


class AttributeHandler(BaseExpressionHandler):
    AST_CLASS = ast.Attribute

    def obj_Material(self, attr):
        self.resmap[0] = mast.Name(f'@{attr.replace("_", "-")}')

    def obj_Liquid(self, attr):
        self.resmap[0] = mast.Name(f'@{attr.replace("_", "-")}')

    def obj_Property(self, attr):
        self.resmap[0] = mast.Name(f'@{attr.replace("_", "-")}')

    def obj_UnitType(self, attr):
        self.resmap[0] = mast.Name(f'@{attr.replace("_", "-")}')

    def obj_BlockFlag(self, attr):
        self.resmap[0] = mast.Name(attr)

    def obj_Block(self, attr):
        self.resmap[0] = mast.Name(f'@{attr.replace("_", "-")}')

    def obj_Target(self, attr):
        self.resmap[0] = mast.Name(attr)

    def obj_Sort(self, attr):
        if attr == 'asc':
            self.resmap[0] = mast.Literal(1)
        elif attr == 'desc':
            self.resmap[0] = mast.Literal(-1)
        else:
            self.resmap[0] = mast.Name(attr)

    def obj_Draw(self, attr):
        raise ValueError(_method_values_permitted.format('Draw'))

    def obj_Control(self, attr):
        raise ValueError(_method_values_permitted.format('Control'))

    def obj_Radar(self, attr):
        raise ValueError(_method_values_permitted.format('Radar'))

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
            self.proc('set', target, value)

    def named_assign(self, target, value):
        _check_assignment_to_reserved(target.id)
        retvals = self.run_trec(value)
        self._assign([mast.Name(target.id)], retvals)

    def tuple_assign(self, target, value):
        assert all(isinstance(n, ast.Name) for n in target.elts)
        tnames = [mast.Name(name.id) for name in target.elts]
        retvals = self.run_trec(value)
        self._assign(tnames, retvals)

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
        t = mast.Name(target.id)
        self.proc(f'op {op}', t, t, operand_val)

    def memory_assign(self, target, op, operand):
        if not isinstance(target.value, ast.Name):
            raise ValueError(f'Unsupported assignment target {target}')
        index_val = self.run_trec_single(_get_ast_slice(target))
        operand_val = self.run_trec_single(operand)
        op_output = mast.Name()
        cell = mast.Name(target.value.id)
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


def transform_expr(expr):
    def trec(expr):
        return transform_expr(expr)

    if type(expr) not in AST_NODE_MAP:
        raise ValueError(f'Unsupported expression {expr}')

    node = AST_NODE_MAP[type(expr)](expr, trec)
    node.handle()
    return node.get_results(), node.pre
