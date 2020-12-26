from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import mast


@dataclass
class FuncDef:
    name: str
    n_args: int
    args: List[mast.Name] = field(
        repr=False, init=False, default=None)
    resmap: Dict[int, mast.Name] = field(init=False, default_factory=dict)
    start_label: mast.Label = field(
        repr=False, init=False, default_factory=mast.Label)
    return_addr: mast.Name = field(
        repr=False, init=False, default_factory=mast.Name)

    def __post_init__(self):
        self.args = [mast.Name() for _ in range(self.n_args)]

    def create_return(self, handler):
        handler.proc('set', mast.Name('@counter'), self.return_addr)


@dataclass
class Scope:
    _parent_scope: Optional['Scope'] = field(repr=False)
    _current_func: Optional[FuncDef] = field(repr=False, default=None)
    _names: dict = field(init=False, default_factory=dict)

    def __getitem__(self, key):
        if key in self._names:
            return self._names[key]
        if self._parent_scope is not None:
            return self._parent_scope.__getitem__(key)
        raise KeyError

    def __setitem__(self, key, value):
        self._names[key] = value

    def __contains__(self, key):
        if key in self._names:
            return True
        if self._parent_scope is not None:
            return self._parent_scope.__contains__(key)
        return False

    @property
    def current_func(self):
        if self._current_func is not None:
            return self._current_func
        if self._parent_scope is not None:
            return self._parent_scope.current_func
        return None
