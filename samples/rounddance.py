from void import (
    M,
    time, allocated_flags,
)


time += 0.5
if time > 360:
    time %= 360

M.unit.bind(M.at.nova)

f = M.unit.flag
if f == 0:
    allocated_flags += 1
    M.unit.setFlag(allocated_flags)
    f = allocated_flags

found, bx, by, building = M.locate.building(M.at.core, False)
if found:
    sx = bx + M.sin(time + f * 30) * 10
    sy = by + M.cos(time + f * 30) * 10

    M.unit.move(sx, sy)
    M.unit.targetPosition(bx, by, True)
