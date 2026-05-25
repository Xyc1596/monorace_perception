from typing import List, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from utils.exp import Exp, DroneState
from utils.gate import Gate
from utils.imports import cv2

class AdaptiveCropper:
    """
    **仅供参考，不使用**
    ## 自适应裁剪
    对于每一帧输入图像，选择最合适的一到两个门框作为当前处理目标；<br/>
    然后仅对这些门框所在的预测区域进行裁剪，得到大小固定的子图像
    """

    def __init__(self, exp: Exp) -> None:
        self._exp = exp
        self._OUTPUT_W: int = exp.module.crop_w
        """自适应裁剪图像宽度（像素）"""
        self._OUTPUT_H: int = exp.module.crop_h
        """自适应裁剪图像高度（像素）"""
        self._GATE_OBLIQUE_THRESH: float = exp.module.crop_gate_oblique_thresh
        """自适应裁剪门框倾斜角阈值（rad），视野中倾斜角大于此值的门框会被排除"""

    def _select_gates(self, drone_state: DroneState, gates: List[Gate]) -> List[Gate]:
        """
        选择合适的一到两个门框作为当前处理目标
        Args:
            drone_state (DroneState): 无人机当前状态
            gates (Sequence[Gate]): 所有检测到的门框
        Returns:
            List[Gate]: 选择的门框列表
        """
        gate_inds = np.arange(len(gates))

        # region 1. 丢弃投影中心在图像边界之外的门框
        R_I2C = drone_state.cam_orientation.as_matrix()  # 3x3 旋转矩阵
        T_C_in_I = drone_state.cam_position.reshape(3, 1)  # 相机在世界坐标系中的位置

        # 计算相机坐标系下的门框中心点
        gate_centers = np.array([gate.center for gate in gates])
        gate_centers_C = (R_I2C.T @ (gate_centers.T - T_C_in_I)).T  # (n, 3)

        # 过滤掉相机后方的点
        front_gate_inds = gate_inds[gate_centers_C[:, 2] > 0.0]
        front_gate_centers = gate_centers[front_gate_inds]

        # 将门框中心投影到相机坐标系
        rvec, _ = cv2.Rodrigues(R_I2C.T)
        tvec: NDArray = (-R_I2C.T @ T_C_in_I).reshape(3)
        front_gate_2d_centers = cv2.projectPoints(
            front_gate_centers,
            rvec,
            tvec,
            self._exp.cam.intrinsics,
            self._exp.cam.dist_coeffs,
        )[0].reshape(-1, 2)

        # 过滤掉图像范围外的门框中心
        visible_gate_inds = front_gate_inds[
            (0 <= front_gate_2d_centers[:, 0] <= self._OUTPUT_W) & (0 <= front_gate_2d_centers[:, 1] <= self._OUTPUT_H)
        ]

        # endregion

        # 2. 丢弃倾斜角过大的门框
        valid_oblique_mask = [
            np.arccos(np.clip(np.dot(gates[cast(int, i)].front, drone_state.cam_front), -1.0, 1.0))
            <= self._GATE_OBLIQUE_THRESH
            for i in visible_gate_inds
        ]
        valid_oblique_gate_inds = visible_gate_inds[valid_oblique_mask]

        # 计算所有门框到无人机当前位置的距离
        distances = np.array(
            cast(float, np.linalg.norm(gates[cast(int, i)].center - drone_state.position))
            for i in valid_oblique_gate_inds
        )
        sorted_gate_inds = valid_oblique_gate_inds[np.argsort(distances)]
        return [gates[i] for i in sorted_gate_inds[:2]]

    def crop(self, drone_state: DroneState, gates: Sequence[Gate]) -> NDArray[np.uint8]:
        """
        对当前帧输入图像进行自适应裁剪
        Args:
            drone_state (DroneState): 无人机当前状态
            gates (Sequence[Gate]): 所有检测到的门框
        Returns:
            NDArray[np.uint8]: 裁剪后的子图像
        """
        pass
