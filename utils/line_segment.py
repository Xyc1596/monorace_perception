from typing import Optional, Tuple, Any

import numpy as np
from numpy.typing import NDArray

from utils.imports import cv2


class LineSegment:
    _EPS = 1e-6
    """容差"""

    def __init__(self, p1: NDArray, p2: NDArray):
        """表示一条线段，包含两个端点坐标"""
        self._start: NDArray = p1
        """端点1 (2,)"""
        self._end: NDArray = p2
        """端点2 (2,)"""
        self._vector = self._end - self._start
        self._length: float = float(np.linalg.norm(self._vector))

        norm = np.linalg.norm(self._vector)
        self._direction: NDArray = np.array([0.0, 0.0]) if norm < 1e-6 else self._vector / norm

    def _extend(self, factor: float) -> None:
        """
        将线段两端均匀扩展，使总长度变为原来的 factor 倍（在原对象上修改）
        :param factor: 长度缩放因子（>1）
        """
        if self._length > 1e-6:
            delta = (self._length * factor - self._length) * self._direction / 2.0
            self._start -= delta
            self._end += delta
            self._vector = self._end - self._start
            self._length = float(np.linalg.norm(self._vector))
            norm = np.linalg.norm(self._vector)
            self._direction = np.array([0.0, 0.0]) if norm < 1e-6 else self._vector / norm

    @property
    def start(self) -> NDArray:
        """起点"""
        return self._start

    @property
    def end(self) -> NDArray:
        """终点"""
        return self._end

    @property
    def length(self) -> float:
        return self._length

    @property
    def vector(self) -> NDArray:
        """线段向量（从p1指向p2）"""
        return self._vector

    @property
    def direction(self) -> NDArray:
        """单位方向向量 (从p1指向p2)"""
        return self._direction

    def extend(self, ext_factor: float) -> "LineSegment":
        """
        将线段两端均匀扩展，使总长度变为原来的 factor 倍
        :param ext_factor: 长度缩放因子（>1）
        :return: 扩展后的新线段
        """
        if (old_len := self.length) < 1e-6:
            return LineSegment(self._start, self._end)
        delta = (old_len * ext_factor - old_len) / 2.0
        return LineSegment(self._start - delta * self._direction, self._end + delta * self._direction)

    def _contains_point(self, point: NDArray) -> bool:
        """判断点是否在线段上；线段退化为点时返回 False"""
        p_vec = point - self.start
        if self._length < self._EPS or abs(np.cross(self._vector, p_vec)) > self._EPS:  # 退化或不共线
            return False
        return self._EPS <= np.dot(p_vec, self._vector) / self._length**2 <= 1 + self._EPS  # 投影是否在两端点之间

    def intersection(self, other: "LineSegment") -> Optional[NDArray]:
        """
        计算与另一条线段交点
        :param other: 另一条线段
        :return: 交点；无交点或共线返回 None
        """
        # 线段退化为点
        if self._length < self._EPS or other._length < self._EPS:
            return None
        
        # 通用情形：参数方程
        # 求解方程：self.start + t * self.direction = other.start + u * other.direction（t、u为标量）
        # 二维叉积为标量 x1*y2-x2*y1
        if abs(cross := np.cross(self._vector, other._vector)) > self._EPS:
            # 不平行：t * self.direction - u * other.direction = other.start - self.start
            start_delta = other._start - self._start
            t = np.cross(start_delta, other._vector) / cross
            u = np.cross(start_delta, self._vector) / cross
            if -self._EPS <= t <= 1 + self._EPS and -self._EPS <= u <= 1 + self._EPS:
                return self._start + np.clip(t, 0, 1) * self._vector
        return None
    
    def plot(self, img: cv2.typing.MatLike):
        """绘制测试图像
        - 灰线：线段
        """
        cv2.line(img, self._start.astype(int).tolist(), self._end.astype(int).tolist(), (127, 127, 127), 1)

    @classmethod
    def of_line(cls, xyxy: Tuple[Any, Any, Any, Any], ext_factor: float = 0.0):
        """
        从LSD检测结果创建线段
        :param xyxy: x1, y1, x2, y2
        :param ext_factor: 长度缩放因子，<=0时跳过该步骤
        :return: 线段对象
        """
        inst = cls(np.array(xyxy[0:2]), np.array(xyxy[2:4]))
        if ext_factor > 0:
            inst._extend(ext_factor)
        return inst
