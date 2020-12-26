from .mast import Jump, Label


_opts = []


def reg(fn):
    _opts.append(fn)
    return fn


@reg
def remove_unnecessary_jumps(instructions):
    remove = set()
    for index, (a, b) in enumerate(zip(instructions, instructions[1:])):
        if isinstance(a, Jump) and isinstance(b, Label):
            if a.label is b:
                remove.add(index)
    if not remove:
        return None
    return [ins for i, ins in enumerate(instructions) if i not in remove]


def optimize(instructions):
    applied = 1
    while applied > 0:
        applied = 0
        for o in _opts:
            ni = o(instructions)
            if ni is not None:
                instructions = ni
                applied += 1
    return instructions
