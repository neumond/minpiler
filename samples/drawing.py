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


def posvel(pos, vel):
    pos += vel
    if pos < 0:
        pos = 0
        vel = M.abs(vel)
    if pos > dsize:
        pos = dsize
        vel = -M.abs(vel)
    return pos, vel


def randvel():
    variation = 10
    return M.rand(variation * 2) - variation


if setup is None or switch1.enabled:
    t1x, t1y = M.rand(dsize), M.rand(dsize)
    t2x, t2y = M.rand(dsize), M.rand(dsize)
    t3x, t3y = M.rand(dsize), M.rand(dsize)
    t1vx, t1vy = randvel(), randvel()
    t2vx, t2vy = randvel(), randvel()
    t3vx, t3vy = randvel(), randvel()
    setup = True
else:
    t1x, t1vx = posvel(t1x, t1vx)
    t1y, t1vy = posvel(t1y, t1vy)
    t2x, t2vx = posvel(t2x, t2vx)
    t2y, t2vy = posvel(t2y, t2vy)
    t3x, t3vx = posvel(t3x, t3vx)
    t3y, t3vy = posvel(t3y, t3vy)


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
