from void import (
    M,
    scatter1, sorter1, switch1, switch2,
    time,
)

sorter1.configure(M.at.copper if switch1.enabled else M.at.lead)

time += 3
if time > 360:
    time %= 360

sx = scatter1.x + M.sin(time) * 10
sy = scatter1.y + M.cos(time) * 10
scatter1.targetPosition(sx, sy, switch2.enabled)
