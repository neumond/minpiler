import pytest

from minpiler import cmdline, emu, mparse, test_utils


# test_transform_expr('Material.copper')
# test_transform_expr('exit()')
# test_transform_expr('1 >= a > 3')
# test_transform_expr('True and True or False and 3')


# test_transform_statement("""
# if a > 3:
#     print('Yes')
# elif a - b:
#     print('Maybe')
# else:
#     print('No')
# """)
# exit()


PROGRAMS = """
print("hello\\n")
print(2 + 2, "\\n")
print(2 + 2 * 2 + 8 + 6 * 9 * 3, "\\n")
print(-5, "\\n")
print(+5, "\\n")
print(max(min(2, 8), 3 + 3), "\\n")
print("line", 1, 2, print(), 3)
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
print(None is None)
--------------
10101
==============
b = 7
print(4 and 8 and b, "\\n")
print(b and None, "\\n")
print(b and None and True, "\\n")
print(False and True)
--------------
7
null
null
0
==============
b = 9
print(None or 0 or b, "\\n")
print(None or 0, "\\n")
print(0 or None, "\\n")
print(None or 0 or 3, "\\n")
print(None or 0 or 3 or 5)
--------------
9
0
null
3
3
==============
print(abs(-6), ' ', sin(90))
--------------
6 1
==============
i = 0
while i < 5:
    print(i)
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
print(pos, ' ', vel)
--------------
-10 20
==============
"""


@pytest.mark.parametrize('code,result', test_utils.parse_programs(PROGRAMS))
def test_programs(code, result):
    print(cmdline.py_to_mind(code))
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
exit()
--------------
end
==============
print(GetLink(4))
--------------
getlink _r1 4
print _r1
==============
Control.setEnabled(conveyor1, True)
Control.targetPosition(duo1, 20, 30, True)
Control.targetObject(duo2, enemy1, False)
Control.configure(unloader1, Material.copper)
--------------
control enabled conveyor1 true
control shoot duo1 20 30 true
control shootp duo2 enemy1 false
control configure unloader1 @copper
==============
print(Radar(
    duo1, Target.enemy, Target.flying, Target.any,
    Sort.distance, Sort.asc))
--------------
radar enemy flying any distance duo1 1 _r1
print _r1
==============
print(Sensor(duo1, Property.health))
--------------
sensor _r1 duo1 @health
print _r1
==============
UnitBind(UnitType.poly)
print(UnitRadar(
    Target.enemy, Target.ground, Target.any,
    Sort.health, Sort.desc))
--------------
ubind @poly
uradar enemy ground any health turret1 -1 _r1
print _r1
==============
found, x, y, building = LocateBuilding(BlockFlag.core, True)
found, x, y = LocateOre(Material.lead)
found, x, y, building = LocateSpawn()
found, x, y, building = LocateDamaged()
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
UnitControl.stop()
UnitControl.move(10, 20)
UnitControl.approach(10, 20, 50)
UnitControl.boost(True)
UnitControl.pathfind()
UnitControl.target(10, 20, True)
UnitControl.targetp(unit, True)
UnitControl.itemDrop(unit, 100)
UnitControl.itemTake(unit, Material.lead, 100)
UnitControl.payDrop()
UnitControl.payTake(10)
UnitControl.mine(10, 20)
UnitControl.flag(5)
UnitControl.build(10, 20, Block.conveyor, 270, Material.lead)
btype, unit = UnitControl.getBlock(10, 20)
result = UnitControl.within(10, 20, 50)
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
