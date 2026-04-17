from typing import Optional, Tuple, Any

import numpy as np
from numpy.typing import NDArray


class LineSegment:
    def __init__(self, p1: NDArray, p2: NDArray):
        """表示一条线段，包含两个端点坐标"""
        self.__start: NDArray = p1
        """端点1 (2,)"""
        self.__end: NDArray = p2
        """端点2 (2,)"""
        self.__vector = self.__end - self.__start
        self.__length: float = float(np.linalg.norm(self.__vector))

        norm = np.linalg.norm(self.__vector)
        self.__direction: NDArray = np.array([0., 0.]) if norm < 1e-6 else self.__vector / norm

    def _extend(self, factor: float) -> None:
        """
        将线段两端均匀扩展，使总长度变为原来的 factor 倍（在原对象上修改）
        :param factor: 长度缩放因子（>1）
        """
        if self.__length > 1e-6:
            delta = (self.__length * factor - self.__length) * self.__direction / 2.0
            self.__start -= delta
            self.__end += delta

    @property
    def start(self) -> NDArray:
        """起点"""
        return self.__start

    @property
    def end(self) -> NDArray:
        """终点"""
        return self.__end

    @property
    def length(self) -> float:
        return self.__length

    @property
    def vector(self) -> NDArray:
        """线段向量（从p1指向p2）"""
        return self.__vector

    @property
    def direction(self) -> NDArray:
        """单位方向向量 (从p1指向p2)"""
        return self.__direction

    def extend(self, ext_factor: float) -> 'LineSegment':
        """
        将线段两端均匀扩展，使总长度变为原来的 factor 倍
        :param ext_factor: 长度缩放因子（>1）
        :return: 扩展后的新线段
        """
        if (old_len := self.length) < 1e-6:
            return LineSegment(self.__start, self.__end)
        delta = (old_len * ext_factor - old_len) / 2.0
        return LineSegment(self.__start - delta * self.__direction, self.__end + delta * self.__direction)

    def intersection(self, other: 'LineSegment') -> Optional[NDArray]:
        """
        计算与另一条线段交点
        :param other: 另一条线段
        :return: 交点；无交点或共线返回 None
        """
        # 求解方程：self.start + t * self.direction = other.start + u * other.direction（t、u为标量）
        # 二维叉积为标量 x1*y2-x2*y1
        if abs(cross := np.cross(self.__vector, other.__vector)) > 1e-6:
            # 不平行：t * self.direction - u * other.direction = other.start - self.start
            start_delta = other.__start - self.__start
            t = np.cross(start_delta, self.__direction) / cross
            u = np.cross(start_delta, other.__direction) / cross
            if -1e-6 <= t <= 1 + 1e-6 and -1e-6 <= u <= 1 + 1e-6:
                return self.__start + np.clip(t, 0, 1) * self.__direction
        return None

    @classmethod
    def of_line(cls, xyxy: Tuple[Any, Any, Any, Any], ext_factor: float = 0.):
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
