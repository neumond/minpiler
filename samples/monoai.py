from void import (
    M,
    message1, sorter1, vault1, container1,
    last_ore,
)

container = vault1 or container1
if not container:
    M.print('No linked container')
    message1.printFlush()
    M.exit()
core_x = container.x
core_y = container.y

if not sorter1.config:
    M.print('Choose ore')
    message1.printFlush()
    M.exit()

M.unit.bind(M.at.mega)

if last_ore != sorter1.config:
    if M.unit.within(core_x, core_y, 5):
        found, ore_x, ore_y = M.locate.ore(sorter1.config)
        if found:
            last_ore = sorter1.config
        else:
            M.print('Can\'t locate ore')
            message1.printFlush()
        M.exit()
    else:
        M.print('Locating ore...')
        message1.printFlush()
        M.unit.approach(core_x, core_y, 5)
    M.exit()

M.print('Gathering...')
message1.printFlush()

dst_core = M.dst(M.unit.x - core_x, M.unit.y - core_y)
dst_ore = M.dst(M.unit.x - ore_x, M.unit.y - ore_y)

closer_to_core = dst_core < dst_ore

items = M.unit.totalItems
cap = M.unit.itemCapacity

full = items == cap
some = 0 < items < cap

if full or (some and closer_to_core):
    if M.unit.within(core_x, core_y, 5):
        M.unit.itemDrop(container, M.unit.totalItems)
    else:
        M.unit.approach(core_x, core_y, 5)
else:
    if M.unit.within(ore_x, ore_y, 5):
        M.unit.mine(ore_x, ore_y)
    else:
        M.unit.approach(ore_x, ore_y, 5)
