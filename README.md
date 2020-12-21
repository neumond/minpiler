# Minpiler

Allows easier control over Mindustry microprocessors.
You can write subset of Python instead assembler-like Mindustry instructions.

**Warning**: Everything that requires dynamic memory doesn't work:
any data structure (list, dict, etc), classes, functions, closures,
recursion. Additionally you can't use imports or any Python builtins
(input, eval, Exception, etc). Explanation below.

## Rationale

Mindustry microprocessors are quite nice and easy to understand.
They have explicit limit of instructions per unit of time.
They can operate on integers, floats, use constant strings and fire
several commands that affect the game.

Here is an example program that switches unloader output
between lead and titanium depending on time:

```
read time cell1 0
op add time time 1
jump 4 lessThan time 300
set time 0
write time cell1 0
jump 8 greaterThan time 200
control configure unloader1 @titanium
end
control configure unloader1 @lead
```

This program is fairly simple, but if you insert any instruction in
the middle of program, you have to rewrite many `jump` addresses.

What if you want to calculate some formula, let's say

```
2 + a * 2 + b + c * d * 3
```

You'll have to write:

```
op mul _r1 a 2
op add _r2 2 _r1
op add _r3 _r2 b
op mul _r4 c d
op mul _r5 _r4 3
op add result _r3 _r5
```

It becomes boring, error-prone and quite hard to modify.

What if you could write the following instead:

```python
time = cell1[0]
time += 1
if time >= 300:
    time = 0
cell1[0] = time
if time > 200:
    Control.configure(unloader1, Material.lead)
else:
    Control.configure(unloader1, Material.titanium)
```

```python
result = 2 + a * 2 + b + c * d * 3
```

It's possible with minpiler!

1. Write your code in subset of Python
2. Compile it into Mindustry instructions
3. Copy the result into clipboard
4. Import code right into running game

So far so good, looks like you can just write Python code
and then import it into the game. As a good Pythonista you're eager to
use your knowledge of closures, classes and decorators to program
Mindustry controllers, but unfortunately I have to stop you there.
You won't be able to use most of the Python. All the restrictions
arise from Mindustry processor architecture:

1. There's no data structures, only scalar values are allowed. The only exception are Memory cells that behave somewhat like dicts.
2. You can't access variables indirectly, there're no pointers, no `getattr`.
3. Subsequence of former, it's impossible to implement dynamic memory/stack (at least without Memory cells). This makes lists, iterators, classes, closures and other things impossible.
4. Jumps can target only fixed locations, there's no indirection using variables. This makes function call stack implementation nearly impossible, you can't return to arbitrary address.
5. Set of builtins is very restricted, you can only call what you have available in game (print, printflush, draw, etc)

Anyway, I hope Minpiler will be handy enough for you to have lots of fun playing Mindustry.

## How to use

```sh
pip install minpiler
```

On Linux systems you can write your code in .py files, then run

```sh
python -m minpiler yourfile.py | xclip -selection clipboard
```

then, open processor building, press Edit button, Import from Clipboard.

It's possible to read source from stdin:

```sh
cat yourfile.py | python -m minpiler -
```

## Compatibility

Currently, only Python 3.8 and Mindustry 121 have been tested.

Some features and autotests are under developement.
