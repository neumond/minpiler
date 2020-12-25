from void import (
    M,
    switch1, vault1,
)


container = vault1


M.unit.bind(M.at.mega)


if switch1.enabled:
    M.unit.setFlag(0)

    core_x = container.x
    core_y = container.y

    ore_x = 56
    ore_y = 140

    moving_home = 20
    at_home = 21
    moving_ore = 22
    at_ore = 23

    M.exit()


if M.unit.flag == moving_home:
    if M.unit.within(core_x, core_y, 5):
        M.unit.setFlag(at_home)
    else:
        M.unit.approach(core_x, core_y, 5)
elif M.unit.flag == at_home:
    if M.unit.totalItems > 0:
        M.unit.itemDrop(container, M.unit.totalItems)
    M.unit.setFlag(moving_ore)
elif M.unit.flag == moving_ore:
    if M.unit.within(ore_x, ore_y, 5):
        M.unit.setFlag(at_ore)
    else:
        M.unit.approach(ore_x, ore_y, 5)
elif M.unit.flag == at_ore:
    if M.unit.totalItems >= M.unit.itemCapacity:
        M.unit.setFlag(moving_home)
    else:
        M.unit.mine(ore_x, ore_y)
else:
    M.unit.setFlag(moving_home)
