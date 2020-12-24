# from void import M, scatter1, sorter1


# if switch1.enabled:
#     sorter1.configure(M.copper)
# else:



if Sensor(switch1, Property.enabled):
    Control.configure(sorter1, Material.copper)
else:
    Control.configure(sorter1, Material.lead)


time += 3
if time > 360:
    time %= 360


sx = Sensor(scatter1, Property.x) + sin(time) * 10
sy = Sensor(scatter1, Property.y) + cos(time) * 10


# TODO: this could be
# scatter1.x
# scatter1.targetPosition(sx, sy, switch2.enabled)


Control.targetPosition(scatter1, sx, sy, Sensor(switch2, Property.enabled))
