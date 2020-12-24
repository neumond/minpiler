if Sensor(switch1, Property.enabled):
    Control.configure(sorter1, Material.copper)
else:
    Control.configure(sorter1, Material.lead)


time += 3
if time > 360:
    time %= 360


sx = Sensor(scatter1, Property.x) + sin(time) * 10
sy = Sensor(scatter1, Property.y) + cos(time) * 10


Control.targetPosition(scatter1, sx, sy, Sensor(switch2, Property.enabled))
