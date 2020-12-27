from void import (
    M,
    unloader1, unloader2, unloader3, unloader4, unloader5, unloader6,
    container1, container2, container3, container4, container5, container6,
    container7,
    conveyor1, conveyor2, conveyor3,
)


def balance(container, unloader):
    lead = container.lead / 50
    sand = container.sand / 50
    coal = container.coal / 25

    if M.min(M.min(sand, coal), lead) >= 1:
        unloader.configure(M.at.spore_pod)
        return

    if lead < M.min(sand, coal):
        unloader.configure(M.at.lead)
    elif sand < coal:
        unloader.configure(M.at.sand)
    else:
        unloader.configure(M.at.coal)


balance(container1, unloader1)
balance(container2, unloader2)
balance(container3, unloader3)
balance(container4, unloader4)
balance(container5, unloader5)
balance(container6, unloader6)

conveyor1.setEnabled(container7.sand < 200)
conveyor2.setEnabled(container7.lead < 200)
conveyor3.setEnabled(container7.coal < 200)
