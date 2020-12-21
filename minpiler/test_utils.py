def parse_programs(pool):
    progs = []

    code = []
    output = []
    reading_code = True

    def flush():
        nonlocal reading_code
        if code or output:
            progs.append((
                '\n'.join(code),
                '\n'.join(output),
            ))
        code.clear()
        output.clear()
        reading_code = True

    for line in pool.strip().splitlines():
        if line.startswith('==='):
            flush()
        elif line.startswith('---'):
            reading_code = False
        else:
            if reading_code:
                code.append(line)
            else:
                output.append(line)
    flush()

    return progs
