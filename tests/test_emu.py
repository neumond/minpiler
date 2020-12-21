from copy import deepcopy

import pytest

from minpiler import emu, mparse, test_utils


@pytest.mark.parametrize('code,state,result', [
    ('op add r a 2', {}, '2'),
    ('op add r a 2', {'a': 2}, '4'),
    ('op add r a 2', {'a': -1}, '1'),
    ('set r 3', {}, '3'),
    ('set r "test"', {}, 'test'),
    ('read r cell1 8', {'cell1': {8: 9.}}, '9'),
    ('read r cell1 400', {'cell1': {8: 9.}}, '0'),
    ('read r cell56 4', {}, '0'),
    ('op not r 0', {}, '-1'),
    ('op not r 2', {}, '-3'),
    ('op or r 4 2', {}, '6'),
    ('op and r 7 3', {}, '3'),
    ('op land r 7 3', {}, '1'),
    ('op sub r 7 xxx', {}, '7'),
    ('op sub r 7 null', {}, '7'),
    ('op sub r 7 true', {}, '6'),
    ('op div r 7 0', {}, '0'),
    ('op idiv r 7 0', {}, '0'),
    ('op pow r 2 2', {}, '4'),
    # ('op abs r -7', {}, '7'),
])
def test_functions(code, state, result):
    ast = mparse.parse(f"{code}\nprint r")
    assert emu.execute(ast, deepcopy(state)) == result


PROGRAMS = """
print "hello"
--------------
hello
==============
set c true
jump 3 equal c true
print "impossible"
print "end"
--------------
end
==============
set c 8
set index 10
write c cell1 index
read v cell1 index
print v
--------------
8
==============
print 1
end
print 2
--------------
1
==============
"""


@pytest.mark.parametrize('code,result', test_utils.parse_programs(PROGRAMS))
def test_programs(code, result):
    ast = mparse.parse(code)
    assert emu.execute(ast, {
        'cell1': {},
    }) == result
