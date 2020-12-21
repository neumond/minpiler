import pytest

from minpiler import cmdline, emu, mparse, test_utils


PROGRAMS = """
print("hello")
--------------
hello
==============
a = -1
a += 6
if a > 0:
    print("a > 0")
else:
    print("a <= 0")
--------------
a > 0
==============
print(1)
exit()
print(2)
--------------
1
==============
print("Hi ", 2 + min(2, 8) * 2)
--------------
Hi 6
==============
cell1[8] = 9
index = 8
cell1[index] += 3
print(cell1[index - 2 + 2])
--------------
12
==============
b = 6
print(4 <= b <= 8)
--------------
1
==============
b = 7
print(4 and 8 and b)
--------------
7
==============
b = 9
print(False or 0 or b)
--------------
9
==============
"""


@pytest.mark.parametrize('code,result', test_utils.parse_programs(PROGRAMS))
def test_programs(code, result):
    ast = mparse.parse(cmdline.py_to_mind(code))
    assert emu.execute(ast, {
        'cell1': {},
    }) == result
