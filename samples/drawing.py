from void import (
    M,
    switch1, display1,
    setup, time,
)

# large display is 176 x 176
dsize = 176

M.draw.clear(0, 0, 0)
M.draw.stroke(5)
M.draw.color(255, 0, 0, 255)

if setup is None or switch1.enabled:
    t1x = M.rand(dsize)
    t1y = M.rand(dsize)
    t2x = M.rand(dsize)
    t2y = M.rand(dsize)
    t3x = M.rand(dsize)
    t3y = M.rand(dsize)
    variation = 10
    t1vx = M.rand(variation * 2) - variation
    t1vy = M.rand(variation * 2) - variation
    t2vx = M.rand(variation * 2) - variation
    t2vy = M.rand(variation * 2) - variation
    t3vx = M.rand(variation * 2) - variation
    t3vy = M.rand(variation * 2) - variation
    setup = True
else:
    pos = t1x
    vel = t1vx

    pos += vel
    if pos < 0:
        pos = 0
        vel = M.abs(vel)
    if pos > dsize:
        pos = dsize
        vel = -M.abs(vel)

    t1x = pos
    t1vx = vel

    pos = t2x
    vel = t2vx

    pos += vel
    if pos < 0:
        pos = 0
        vel = M.abs(vel)
    if pos > dsize:
        pos = dsize
        vel = -M.abs(vel)

    t2x = pos
    t2vx = vel

    pos = t3x
    vel = t3vx

    pos += vel
    if pos < 0:
        pos = 0
        vel = M.abs(vel)
    if pos > dsize:
        pos = dsize
        vel = -M.abs(vel)

    t3x = pos
    t3vx = vel

    pos = t1y
    vel = t1vy

    pos += vel
    if pos < 0:
        pos = 0
        vel = M.abs(vel)
    if pos > dsize:
        pos = dsize
        vel = -M.abs(vel)

    t1y = pos
    t1vy = vel

    pos = t2y
    vel = t2vy

    pos += vel
    if pos < 0:
        pos = 0
        vel = M.abs(vel)
    if pos > dsize:
        pos = dsize
        vel = -M.abs(vel)

    t2y = pos
    t2vy = vel

    pos = t3y
    vel = t3vy

    pos += vel
    if pos < 0:
        pos = 0
        vel = M.abs(vel)
    if pos > dsize:
        pos = dsize
        vel = -M.abs(vel)

    t3y = pos
    t3vy = vel


M.draw.line(t1x, t1y, t2x, t2y)
M.draw.line(t2x, t2y, t3x, t3y)
M.draw.line(t3x, t3y, t1x, t1y)


time += 10
if time > 360:
    time %= 360


M.draw.color(255, 255, 255, 255)
M.draw.image(t1x, t1y, M.at.lead, 30 + M.sin(time) * 10, time)
M.draw.image(t2x, t2y, M.at.metaglass, 30 + M.sin(time + 120) * 10, time)
M.draw.image(t3x, t3y, M.at.thorium, 30 + M.sin(time + 240) * 10, time)


display1.drawFlush()
