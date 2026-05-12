from enum import Enum
from typing import Dict, Tuple

from loguru import logger


class CornerDescriptor(Enum):
    LT = (0, (True, False, False, False))
    RT = (1, (False, True, False, False))
    RB = (2, (False, False, True, False))
    LB = (3, (True, False, False, True))
    UNDEFINED = (4, (False, False, False, False))

    def __init__(self, idx: int, tup: Tuple[bool, bool, bool, bool]):
        self._idx = idx
        self._tup = tup

    @property
    def descriptor(self):
        return self._tup

    def __eq__(self, other: object) -> bool:
        return self._idx == other._idx if isinstance(other, CornerDescriptor) else super().__eq__(other)

    @classmethod
    def of_tuple(cls, tup: Tuple[bool, bool, bool, bool]) -> "CornerDescriptor":
        if tup in _TUPLE_MAP:
            return _TUPLE_MAP[tup]
        else:
            logger.warning(f"Undefined corner descriptor: {tup}")
            return CornerDescriptor.UNDEFINED


_TUPLE_MAP: Dict[Tuple[bool, bool, bool, bool], CornerDescriptor] = {
    (True, False, False, False): CornerDescriptor.LT,
    (False, True, False, False): CornerDescriptor.RT,
    (False, False, True, False): CornerDescriptor.RB,
    (False, False, False, True): CornerDescriptor.LB,
}
