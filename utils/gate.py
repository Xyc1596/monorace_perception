"""
门框预定义
"""

from typing import Dict, List, Sequence

import numpy as np
from numpy.typing import NDArray

from utils import Corner2D, CornerDescriptor
from utils.exp import DroneState, Exp
from utils.imports import cv2
from cv2.typing import MatLike


class Gate:
    CORNER_ORDER: List[CornerDescriptor] = [
        CornerDescriptor.LTO,
        CornerDescriptor.RTO,
        CornerDescriptor.RBO,
        CornerDescriptor.LBO,
        CornerDescriptor.LTI,
        CornerDescriptor.RTI,
        CornerDescriptor.RBI,
        CornerDescriptor.LBI,
    ]

    def __init__(
        self,
        gate_id: int,
        gate_pose: Sequence[float],
        gate_outer_half_w: float,
        gate_outer_half_h: float,
        gate_inner_half_w: float,
        gate_inner_half_h: float,
    ):
        """门框定义

        Args:
            gate_id (int): 门框编号
            gate_pose (Sequence[float]): 门框坐标（m）和偏航（rad） [x, y, z, yaw]
            gate_outer_half_w (float): 门框外部半宽度（米）
            gate_outer_half_h (float): 门框外部半高度（米）
            gate_inner_half_w (float): 门框内部半宽度（米）
            gate_inner_half_h (float): 门框内部半高度（米）
        """
        yaw = gate_pose[3]

        self.gate_id: int = gate_id
        """门框编号"""
        self.center: NDArray[np.float_] = np.array(gate_pose[:3])
        """门框前表面中心坐标"""
        self._front: NDArray[np.float_] = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        """门框方向向量（前表面法向量，指向无人机穿过方向）"""
        self._left: NDArray[np.float_] = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        """门框左向量"""
        self._up: NDArray[np.float_] = np.array([0.0, 0.0, 1.0])
        """门框上向量"""

        self.corner_3d_points: Dict[CornerDescriptor, NDArray[np.float_]] = dict(
            zip(
                self.CORNER_ORDER,
                [
                    self.center + gate_outer_half_w * self._left + gate_outer_half_h * self._up,
                    self.center - gate_outer_half_w * self._left + gate_outer_half_h * self._up,
                    self.center - gate_outer_half_w * self._left - gate_outer_half_h * self._up,
                    self.center + gate_outer_half_w * self._left - gate_outer_half_h * self._up,
                    self.center + gate_inner_half_w * self._left + gate_inner_half_h * self._up,
                    self.center - gate_inner_half_w * self._left + gate_inner_half_h * self._up,
                    self.center - gate_inner_half_w * self._left - gate_inner_half_h * self._up,
                    self.center + gate_inner_half_w * self._left - gate_inner_half_h * self._up,
                ],
            )
        )
        """门框角点3D坐标"""


class GateManager:
    def __init__(self, exp: Exp):
        """所有门框

        Args:
            gate_pose (Sequence[float]): 门框前表面中心坐标及偏航角 (x, y, z, yaw) 序列
        """
        self._exp = exp
        self._all_gates: List[Gate] = []
        """所有门框"""

        all_3d_corners_list: List[NDArray] = []
        corner_to_gate_id_lookup_list: List[int] = []

        for gate_id, pose in enumerate(self._exp.gate.poses):
            gate = Gate(
                gate_id,
                pose,
                self._exp.gate.outer_half_w,
                self._exp.gate.outer_half_h,
                self._exp.gate.inner_half_w,
                self._exp.gate.inner_half_h,
            )
            self._all_gates.append(gate)
            all_3d_corners_list.extend(gate.corner_3d_points.values())
            corner_to_gate_id_lookup_list.extend([gate_id] * len(gate.corner_3d_points))

        self._all_3d_corners: NDArray = np.array(all_3d_corners_list)
        """所有门框角点3D坐标 (n_gates * 8, 3)"""
        self._corner_gate_id_lookup: NDArray = np.array(corner_to_gate_id_lookup_list)
        """角点索引值到门框编号的映射 (n_gates * 8,)"""
        self._corner_descriptor_lookup: List[CornerDescriptor] = Gate.CORNER_ORDER * len(self._all_gates)
        """角点索引值到描述子的映射 (n_gates * 8,)"""
        
    def __len__(self) -> int:
        return len(self._all_gates)
    
    def render_all_corners(self, drone_state: DroneState) -> List[Corner2D]:
        """将可见门框角点映射到相机画面
        Args:
            drone_state (DroneState): 无人机状态
        Returns:
            List[Corner2D]: 门框角点
        """
        # 获取相机位姿
        # cam_orientation 是 I2C（世界坐标系到相机坐标系的旋转）
        R_I2C = drone_state.cam_orientation.as_matrix()  # 3x3 旋转矩阵
        T_C_in_I = drone_state.cam_position.reshape(3, 1)  # 相机在世界坐标系中的位置

        # 计算相机坐标系下的点：X_cam = R_I2C^T @ (X_world - T_C_in_I)
        corners_cam = (R_I2C.T @ (self._all_3d_corners.T - T_C_in_I)).T  # (n, 3)

        # 过滤掉相机后方的点（z <= 0 表示在相机后方或在相机平面上）
        visible_mask = corners_cam[:, 2] > 0
        visible_indices = np.where(visible_mask)[0]

        if len(visible_indices) == 0:
            return []

        visible_corners_3d = self._all_3d_corners[visible_mask]
        visible_gate_ids = self._corner_gate_id_lookup[visible_mask]

        # 使用 OpenCV projectPoints 进行投影
        # projectPoints 执行: X_img = K @ (R @ X_world + t)
        # 我们需要: R = R_I2C^T, t = -R_I2C^T @ T_C_in_I
        rvec, _ = cv2.Rodrigues(R_I2C.T)
        tvec: NDArray = (-R_I2C.T @ T_C_in_I).reshape(3)

        corners_2d, _ = cv2.projectPoints(
            objectPoints=visible_corners_3d,
            rvec=rvec,
            tvec=tvec,
            cameraMatrix=self._exp.cam.intrinsics,
            distCoeffs=self._exp.cam.dist_coeffs,
        )
        corners_2d = corners_2d.reshape(-1, 2).astype(np.int_)

        # 过滤掉图像范围外的点
        result: List[Corner2D] = []
        img_w = self._exp.cam.img_w
        img_h = self._exp.cam.img_h
        for point, gate_id, orig_idx in zip(corners_2d, visible_gate_ids, visible_indices):
            # 检查点是否在图像范围内
            if 0 <= point[0] < img_w and 0 <= point[1] < img_h:
                # 根据原始索引确定描述子（每个门框8个角点）
                descriptor = Gate.CORNER_ORDER[orig_idx % 8]
                result.append(Corner2D(point, descriptor, gate_id))

        return result
