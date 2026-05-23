from typing import List

import numpy as np
from numpy.typing import NDArray

from .corner_descriptor import CornerDescriptor
from .line_segment import LineSegment
from utils.imports import cv2


class Corner2D:
    def __init__(self, point: NDArray[np.int_], descriptor: CornerDescriptor, gate_id: int = -1):
        """2D图像内门框角点

        Args:
            point (NDArray[np.int_]): 坐标
            descriptor (CornerDescriptor): 描述子，表示角点位于门框的哪个角
            gate_id (int): 所属门框编号，-1 表示未知
        """
        self._point = point
        self._descriptor = descriptor
        self._gate_id = gate_id

    @property
    def point(self):
        """角点坐标 (2,)"""
        return self._point

    @property
    def descriptor(self):
        """角点描述子，表示角点位于门框的哪个角"""
        return self._descriptor

    def distance_to(self, other: "Corner2D") -> float:
        return float(np.linalg.norm(self._point - other._point))

    def descriptor_matched(self, other: "Corner2D") -> bool:
        return self._descriptor == other._descriptor

    def is_from_same_gate(self, other: "Corner2D") -> bool:
        return self._gate_id >= 0 and self._gate_id == other._gate_id

    def __str__(self) -> str:
        return f"Corner{{point: {self._point}, descriptor: {self._descriptor}, gate_id: {self._gate_id}}}"

    def plot(self, img: cv2.typing.MatLike):
        """绘制测试图像
        - 红点：角点坐标
        """
        cv2.circle(img, self._point.tolist(), 1, (0, 0, 255), -1)


class Corner2DFromMask(Corner2D):
    _OFFSET_SIGN = (
        (-1, -1),  # LT
        (1, -1),  # RT
        (1, 1),  # RB
        (-1, 1),  # LB
    )

    def __init__(self, point: NDArray[np.int_], line1: LineSegment, line2: LineSegment, mask: NDArray, offset: int = 5):
        """
        从分割掩码提取的门框角点，包含坐标、生成它的两条线段和描述子

        Args:
            point (NDArray[np.int_]): 候选角点坐标
            line1 (LineSegment): 生成角点的线段1
            line2 (LineSegment): 生成角点的线段2
            mask (NDArray): 二值化掩码图像（0/255）
            offset (int): 描述子采样偏移像素数
        """
        self._line1: LineSegment = line1
        self._line2: LineSegment = line2
        self._descriptor_points: List[NDArray[np.int_]] = []

        # 提取角点描述子
        d1 = line1.direction
        d2 = line2.direction
        # 如果方向向量为零向量（线段退化），使用默认方向
        if np.linalg.norm(d1) < 1e-6:
            d1 = np.array([1.0, 0.0])
        if np.linalg.norm(d2) < 1e-6:
            d2 = np.array([0.0, 1.0])

        h, w = mask.shape[:2]
        descriptor: list[bool] = []
        for off in self._OFFSET_SIGN:
            desc_point = (point + off[0] * offset * d1 + off[1] * offset * d2).astype(int)
            px = int(desc_point[0])
            py = int(desc_point[1])
            self._descriptor_points.append(np.array([px, py]))
            if 0 <= px < w and 0 <= py < h:
                descriptor.append(mask[py, px] > 0)
            else:
                descriptor.append(False)

        super().__init__(point, CornerDescriptor.of_bool_sequence(tuple(descriptor)))  # type: ignore

    def match_prior(self, prior: Corner2D):
        """与先验交点关联，设置所属门框编号"""
        if self._gate_id >= 0:
            raise ValueError(f"角点已与门框 {self._gate_id} 匹配")
        self._gate_id = prior._gate_id

    def plot(self, img: cv2.typing.MatLike):
        """绘制测试图像
        - 绿点：角点坐标
        - 蓝点：描述子采样点坐标
        """
        cv2.circle(img, self._point.tolist(), 1, (0, 255, 0), -1)
        for dp in self._descriptor_points:
            cv2.circle(img, dp.tolist(), 1, (255, 0, 0), -1)
