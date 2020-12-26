from void import (
    M,
    unloader1, unloader2, unloader3, unloader4,
    container1, container2, container3, container4,
)


def balance(container, unloader):
    if container.lead < 50:
        unloader.configure(M.at.lead)
    elif container.sand < 50:
        unloader.configure(M.at.sand)
    elif container.coal < 25:
        unloader.configure(M.at.coal)
    else:
        unloader.configure(M.at.spore_pod)


balance(container1, unloader1)
balance(container2, unloader2)
balance(container3, unloader3)
balance(container4, unloader4)
