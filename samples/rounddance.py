time += 0.5
if time > 360:
    time %= 360

UnitBind(UnitType.nova)

f = Sensor(Material.unit, Property.flag)
if f == 0:
    allocated_flags += 1
    UnitControl.flag(allocated_flags)
    f = allocated_flags

found, bx, by, building = LocateBuilding(BlockFlag.core, False)
if found:
    sx = bx + sin(time + f * 30) * 10
    sy = by + cos(time + f * 30) * 10

    UnitControl.move(sx, sy)
    UnitControl.target(bx, by, False)
    # Control.targetPosition(Material.unit, bx, by, True)

