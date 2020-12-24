print('Hello world!\nHow are you?\n')
print('totalItems: ', Sensor(conveyor1, Property.totalItems), '\n')
print('firstItem: ', Sensor(conveyor1, Property.firstItem), '\n')
print('switch.enabled: ', Sensor(switch1, Property.enabled), '\n')
print('switch.config: ', Sensor(switch1, Property.config), '\n')
print('switch.flag: ', Sensor(switch1, Property.flag), '\n')
print.flush(message1)

# large display is 176 x 176
dsize = 176

Draw.clear(0, 0, 0)
Draw.stroke(5)
Draw.color(255, 0, 0, 255)
# Draw.rect(20, 20, 20, 20)
# Draw.lineRect(0, 0, 176, 176)
# Draw.color(255, 255, 255, 255)
# Draw.line(0, 0, 176, 176)
# Draw.color(255, 255, 0, 255)
# Draw.poly(100, 40, 5, 30, 10)
# Draw.color(255, 0, 255, 255)
# Draw.linePoly(40, 100, 5, 30, 10)


if setup == None or Sensor(switch1, Property.enabled):
    t1x = rand(dsize)
    t1y = rand(dsize)
    t2x = rand(dsize)
    t2y = rand(dsize)
    t3x = rand(dsize)
    t3y = rand(dsize)
    variation = 10
    t1vx = rand(variation * 2) - variation
    t1vy = rand(variation * 2) - variation
    t2vx = rand(variation * 2) - variation
    t2vy = rand(variation * 2) - variation
    t3vx = rand(variation * 2) - variation
    t3vy = rand(variation * 2) - variation
    setup = True
else:
    pos = t1x
    vel = t1vx

    pos += vel
    if pos < 0:
        pos = 0
        vel = abs(vel)
    if pos > dsize:
        pos = dsize
        vel = -abs(vel)

    t1x = pos
    t1vx = vel

    pos = t2x
    vel = t2vx

    pos += vel
    if pos < 0:
        pos = 0
        vel = abs(vel)
    if pos > dsize:
        pos = dsize
        vel = -abs(vel)

    t2x = pos
    t2vx = vel

    pos = t3x
    vel = t3vx

    pos += vel
    if pos < 0:
        pos = 0
        vel = abs(vel)
    if pos > dsize:
        pos = dsize
        vel = -abs(vel)

    t3x = pos
    t3vx = vel

    pos = t1y
    vel = t1vy

    pos += vel
    if pos < 0:
        pos = 0
        vel = abs(vel)
    if pos > dsize:
        pos = dsize
        vel = -abs(vel)

    t1y = pos
    t1vy = vel

    pos = t2y
    vel = t2vy

    pos += vel
    if pos < 0:
        pos = 0
        vel = abs(vel)
    if pos > dsize:
        pos = dsize
        vel = -abs(vel)

    t2y = pos
    t2vy = vel

    pos = t3y
    vel = t3vy

    pos += vel
    if pos < 0:
        pos = 0
        vel = abs(vel)
    if pos > dsize:
        pos = dsize
        vel = -abs(vel)

    t3y = pos
    t3vy = vel


# Draw.triangle(t1x, t1y, t2x, t2y, t3x, t3y)
Draw.line(t1x, t1y, t2x, t2y)
Draw.line(t2x, t2y, t3x, t3y)
Draw.line(t3x, t3y, t1x, t1y)
Draw.color(255, 255, 255, 255)


time += 10
if time > 360:
    time %= 360


Draw.image(t1x, t1y, Material.lead, 30 + sin(time) * 10, time)
Draw.image(t2x, t2y, Material.metaglass, 30 + sin(time + 120) * 10, time)
Draw.image(t3x, t3y, Material.thorium, 30 + sin(time + 240) * 10, time)


Draw.flush(display1)
