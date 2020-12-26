import pytest

from minpiler import cmdline, emu, mparse, test_utils


PROGRAMS = """
M.print("hello\\n")
M.print(2 + 2, "\\n")
M.print(2 + 2 * 2 + 8 + 6 * 9 * 3, "\\n")
M.print(-5, "\\n")
M.print(+5, "\\n")
M.print(M.max(M.min(2, 8), 3 + 3), "\\n")
M.print("line", 1, 2, M.print(), 3)
--------------
hello
4
176
-5
5
6
line12null3
==============
2 + 6
a = -1
a += 6
if a > 0:
    M.print("a > 0")
else:
    M.print("a <= 0")
--------------
a > 0
==============
M.print(1)
M.exit()
M.print(2)
--------------
1
==============
M.print("Hi ", 2 + M.min(2, 8) * 2)
--------------
Hi 6
==============
cell1[8] = 9
index = 8
cell1[index] += 3
M.print(cell1[index - 2 + 2])
--------------
12
==============
b = 6
M.print(4 <= b <= 8)
M.print(b > 8)
M.print(b == 6)
M.print(99 < b)
M.print(None is None)
--------------
10101
==============
b = 7
M.print(4 and 8 and b, "\\n")
M.print(b and None, "\\n")
M.print(b and None and True, "\\n")
M.print(False and True)
--------------
7
null
null
0
==============
b = 9
M.print(None or 0 or b, "\\n")
M.print(None or 0, "\\n")
M.print(0 or None, "\\n")
M.print(None or 0 or 3, "\\n")
M.print(None or 0 or 3 or 5)
--------------
9
0
null
3
3
==============
M.print(M.abs(-6), ' ', M.sin(90))
--------------
6 1
==============
i = 0
while i < 5:
    M.print(i)
    i += 1
--------------
01234
==============
pos = 10
vel = -20
pos += vel
if pos < 0 and vel < 0:
    vel = -vel
if pos > 100 and vel > 0:
    vel = -vel
M.print(pos, ' ', vel)
--------------
-10 20
==============
M.print(True if 2 > 3 else False)
M.print(True if 2 < 3 else False)
--------------
01
==============
a, b = M.abs(-2), 3 + 6
M.print(a, ' ', b)
--------------
2 9
==============
def fn(a):
    M.print(a)
    return a + 1

b = 3
M.print(fn(b + 4))
M.print("9")
--------------
789
==============
a = 4

def fn(a):
    a = 6
    a += 1
    M.print(a, "\\n")

fn(3)
M.print(a)
--------------
7
4
==============
def balance(container):
    M.print("Object ", container, "\\n")
    M.print("Lead ", container.lead)

balance(container1)
--------------
Object container
Lead 300
==============
"""


@pytest.mark.parametrize('code,result', test_utils.parse_programs(PROGRAMS))
def test_programs(code, result):
    print(cmdline.py_to_mind(code))
    ast = mparse.parse(cmdline.py_to_mind(code))
    assert emu.execute(ast, {
        'cell1': {},
        'container1': {'@lead': 300, '__str__': 'container'},
    }) == result


CODEGEN = """
==============
M.draw.clear(1, 2, 3)
M.draw.color(1, 2, 3, 200)
M.draw.stroke(3)
M.draw.line(20, 20, 50, 50)
M.draw.rect(20, 20, 10, 10)
M.draw.lineRect(20, 20, 10, 10)
M.draw.poly(50, 50, 6, 30, 10)
M.draw.linePoly(50, 50, 6, 30, 10)
M.draw.triangle(20, 20, 60, 60, 30, 10)
M.draw.image(50, 50, M.at.copper, 32, 0)
display1.drawFlush()
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
M.print(1)
M.print(b)
M.print('string')
message1.printFlush()
--------------
print 1
print b
print "string"
printflush message1
==============
M.exit()
--------------
end
==============
M.print(M.getLink(4))
--------------
getlink _r1 4
print _r1
==============
conveyor1.setEnabled(True)
duo1.targetPosition(20, 30, True)
duo2.targetObject(enemy1, False)
unloader1.configure(M.at.copper)
--------------
control enabled conveyor1 true
control shoot duo1 20 30 true
control shootp duo2 enemy1 false
control configure unloader1 @copper
==============
M.print(duo1.radar(
    M.at.enemy, M.at.flying, M.at.any,
    M.at.distance, M.at.asc))
--------------
radar enemy flying any distance duo1 1 _r1
print _r1
==============
M.print(duo1.sensor(M.at.health))
M.print(duo1.health)
--------------
sensor _r1 duo1 @health
print _r1
sensor _r2 duo1 @health
print _r2
==============
M.unit.bind(M.at.poly)
M.print(M.unit.radar(
    M.at.enemy, M.at.ground, M.at.any,
    M.at.health, M.at.desc))
--------------
ubind @poly
uradar enemy ground any health turret1 -1 _r1
print _r1
==============
found, x, y, building = M.locate.building(M.at.core, True)
found, x, y = M.locate.ore(M.at.lead)
found, x, y, building = M.locate.spawn()
found, x, y, building = M.locate.damaged()
--------------
ulocate building core true @copper _r1 _r2 _r3 _r4
set found _r3
set x _r1
set y _r2
set building _r4
ulocate ore core true @lead _r5 _r6 _r7 _r8
set found _r7
set x _r5
set y _r6
ulocate spawn core true @copper _r9 _r10 _r11 _r12
set found _r11
set x _r9
set y _r10
set building _r12
ulocate damaged core true @copper _r13 _r14 _r15 _r16
set found _r15
set x _r13
set y _r14
set building _r16
==============
M.unit.stop()
M.unit.move(10, 20)
M.unit.approach(10, 20, 50)
M.unit.boost(True)
M.unit.pathfind()
M.unit.targetPosition(10, 20, True)
M.unit.targetObject(unit, True)
M.unit.itemDrop(unit, 100)
M.unit.itemTake(unit, M.at.lead, 100)
M.unit.payDrop()
M.unit.payTake(10)
M.unit.mine(10, 20)
M.unit.setFlag(5)
M.unit.build(10, 20, M.at.conveyor, 270, M.at.lead)
btype, unit = M.unit.getBlock(10, 20)
result = M.unit.within(10, 20, 50)
--------------
ucontrol stop 0 0 0 0 0
ucontrol move 10 20 0 0 0
ucontrol approach 10 20 50 0 0
ucontrol boost true 0 0 0 0
ucontrol pathfind 0 0 0 0 0
ucontrol target 10 20 true 0 0
ucontrol targetp unit true 0 0 0
ucontrol itemDrop unit 100 0 0 0
ucontrol itemTake unit @lead 100 0 0
ucontrol payDrop 0 0 0 0 0
ucontrol payTake 10 0 0 0 0
ucontrol mine 10 20 0 0 0
ucontrol flag 5 0 0 0 0
ucontrol build 10 20 @conveyor 270 @lead
ucontrol getBlock 10 20 _r1 _r2 0
set btype _r1
set unit _r2
ucontrol within 10 20 50 _r3 0
set result _r3
==============
"""


@pytest.mark.parametrize(
    'pycode,mlogcode', test_utils.parse_programs(CODEGEN))
def test_codegen(pycode, mlogcode):
    assert mlogcode.strip() == cmdline.py_to_mind(pycode).strip()


@pytest.mark.parametrize('pycode', [
    'print = 4',
    'b = print',
    'print + 8',
    'key = GetLink',
    'Draw[8] = True',
])
def test_reserved_name(pycode):
    with pytest.raises(ValueError):
        cmdline.py_to_mind(pycode)
