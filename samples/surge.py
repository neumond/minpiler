from minpiler.typeshed import (
    M, MObject,
    unloader1, unloader2, unloader3,
)


unloader1: MObject
unloader2: MObject
unloader3: MObject


# loads 1 plastanium conveyor using 3 unloaders
# ratio is for surge alloy


period = 10_000  # milliseconds
parts = 4
t = (M.at.time % period) / period * 4

unloader1.configure(M.at.lead)
unloader2.configure(M.at.silicon if t < 1 else M.at.copper)
unloader3.configure(M.at.silicon if t > 2 else M.at.titanium)
