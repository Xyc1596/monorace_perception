from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from utils.line_segment import LineSegment


class Corner:
    def __init__(self, point: NDArray[int], descriptor: Tuple[bool, bool, bool, bool], gate_id: int = -1):
        """
        门框角点
        :param point: 坐标
        :param descriptor: 描述子：左上，右上，右下，左下属于掩码（True）还是背景（False）
        :param gate_id: 所属门框编号，-1 表示未知
        """
        self._point = point
        self._descriptor = descriptor
        self._gate_id = gate_id

    @property
    def point(self):
        """角点坐标 (2,)"""
        return self._point

    def distance_to(self, other: 'Corner') -> float:
        return float(np.linalg.norm(self._point - other._point))

    def descriptor_matched(self, other: 'Corner') -> bool:
        return self._descriptor == other._descriptor


class CornerFromMask(Corner):
    def __init__(
            self,
            point: NDArray[int],
            line1: LineSegment,
            line2: LineSegment,
            mask: NDArray,
            offset: int = 5
    ):
        """
        门框角点，包含坐标、生成它的两条线段和描述子
        :param point: 候选角点坐标
        :param line1: 生成角点的线段1
        :param line2: 生成角点的线段2
        :param mask: 二值化掩码图像（0/255）
        :param offset: 描述子偏移像素数
        """
        self._line1: LineSegment = line1
        self._line2: LineSegment = line2

        # 提取角点描述子
        d1 = line1.direction
        d2 = line2.direction
        # 如果方向向量为零向量（线段退化），使用默认方向
        if np.linalg.norm(d1) < 1e-6:
            d1 = np.array([1.0, 0.0])
        if np.linalg.norm(d2) < 1e-6:
            d2 = np.array([0.0, 1.0])

        h, w = mask.shape[:2]
        descriptor = []
        for off in [offset * d1, -offset * d1, offset * d2, -offset * d2]:
            px: int = point[0] + off[0]
            py: int = point[1] + off[1]
            if 0 <= px < w and 0 <= py < h:
                descriptor.append(mask[py, px] > 0)
            else:
                descriptor.append(False)

        super().__init__(point, tuple(descriptor))  # type: ignore

    def set_gate_id(self, gate_id: int):
        if self._gate_id >= 0:
            raise ValueError(f"角点已与门框 {self._gate_id} 匹配")
        elif gate_id < 0:
            raise ValueError(f"无效的门框编号 {gate_id} （需要：>=0 的整数）")
        else:
            self._gate_id = gate_id
