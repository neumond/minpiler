from void import (
    M,
    unloader1, unloader2, unloader3,
)


# loads 1 plastanium conveyor using 3 unloaders
# ratio is for surge alloy


period = 10_000  # milliseconds
parts = 4
t = (M.at.time % period) / period * 4

unloader1.configure(M.at.lead)
unloader2.configure(M.at.silicon if t < 1 else M.at.copper)
unloader3.configure(M.at.silicon if t > 2 else M.at.titanium)
