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
print(b > 8)
print(b == 6)
print(99 < b)
--------------
1010
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
print(abs(-6), ' ', sin(90))
--------------
6 1
==============
"""


@pytest.mark.parametrize('code,result', test_utils.parse_programs(PROGRAMS))
def test_programs(code, result):
    ast = mparse.parse(cmdline.py_to_mind(code))
    assert emu.execute(ast, {
        'cell1': {},
    }) == result


CODEGEN = """
==============
Draw.clear(1, 2, 3)
Draw.color(1, 2, 3, 200)
Draw.stroke(3)
Draw.line(20, 20, 50, 50)
Draw.rect(20, 20, 10, 10)
Draw.lineRect(20, 20, 10, 10)
Draw.poly(50, 50, 6, 30, 10)
Draw.linePoly(50, 50, 6, 30, 10)
Draw.triangle(20, 20, 60, 60, 30, 10)
Draw.image(50, 50, Material.copper, 32, 0)
Draw.flush(display1)
--------------
draw clear 1 2 3
draw color 1 2 3 200
draw stroke 3
draw line 20 20 50 50
draw rect 20 20 10 10
draw lineRect 20 20 10 10
draw poly 50 50 6 30 10
draw linePoly 50 50 6 30 10
draw triangle 20 20 60 60 30 10
draw image 50 50 @copper 32 0
drawflush display1
==============
print(1)
print(b)
print('string')
print.flush(message1)
--------------
print 1
print b
print "string"
printflush message1
==============
"""


@pytest.mark.parametrize(
    'pycode,mlogcode', test_utils.parse_programs(CODEGEN))
def test_codegen(pycode, mlogcode):
    assert mlogcode.strip() == cmdline.py_to_mind(pycode).strip()
