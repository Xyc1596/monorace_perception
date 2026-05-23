from enum import Enum
from typing import Sequence

from loguru import logger


class CornerDescriptor(Enum):
    """角点描述子，表示角点位于门框的哪个角"""

    UNDEFINED = 0b0000
    """无效"""
    LTO = 0b0010
    """左上角外侧"""
    RTO = 0b0001
    """右上角外侧"""
    RBO = 0b1000
    """右下角外侧"""
    LBO = 0b0100
    """左下角外侧"""
    LTI = 0b1101
    """左上角内侧"""
    RTI = 0b1110
    """右上角内侧"""
    RBI = 0b0111
    """右下角内侧"""
    LBI = 0b1011
    """左下角内侧"""

    def __hash__(self) -> int:
        return self.value

    def __eq__(self, other: object) -> bool:
        if self.value == CornerDescriptor.UNDEFINED.value:
            return False
        return self.value == other.value if isinstance(other, CornerDescriptor) else False

    @classmethod
    def of_bool_sequence(cls, bool_seq: Sequence[bool]) -> "CornerDescriptor":
        """从bool序列创建描述子

        Args:
            bool_seq (Sequence[bool]): 角点周围4个像素属于门框还是背景 (左上，右上，右下，左下)

        Raises:
            ValueError: 序列长度不足4

        Returns:
            CornerDescriptor: 角点描述子
        """
        if (seq_len := len(bool_seq)) < 4:
            raise ValueError(f"bool_seq must have at least 4 elements, but got {seq_len}")
        i = sum((1 << i) for i, b in enumerate(bool_seq[:4]) if b)
        try:
            return cls(i)
        except ValueError:
            logger.warning(f"Undefined corner descriptor: {bool_seq}")
            return CornerDescriptor.UNDEFINED
