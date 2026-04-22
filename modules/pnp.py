"""
PnP位姿估计
"""

from typing import Optional, Tuple, Sequence, List
import typing

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from utils.imports import cv2
from utils.corner import Corner


class PnP:
    def __init__(self, intrinsic_mat: NDArray, dist_coeffs: NDArray):
        self._INTRINSIC_MAT: NDArray = intrinsic_mat
        self._DIST_COEFFS: NDArray = dist_coeffs
        """相机畸变系数"""
        self._RANSAC_THRESH: float = 5.0
        """RANSAC内点阈值（像素）"""
        self._MIN_CORNERS: int = 6
        """使用完整PnP解决方案所需的最小角点数量"""
        self._MIN_DISTANCE: float = 2.0
        """使用完整PnP解决方案的最小门距离（米）"""
        self._MAX_DISTANCE: float = 5.0
        """使用完整PnP解决方案的最大门距离（米）"""

    def solve(self, corners: Sequence[Corner], object_points: NDArray) -> Optional[Tuple[NDArray, NDArray]]:
        """
        求解PnP问题，估计相机位姿
        Args:
            corners (Sequence[Corner]): Corner类型对象组成的序列
            object_points (NDArray): 对应的3D世界点坐标 [N, 3]
        Returns:
            Optional[Tuple[NDArray, NDArray]]: (旋转向量, 平移向量)，若失败则返回None
        """
        if corners is None or len(corners) < 4:
            return None

        if object_points.shape[0] != len(corners):
            return None

        # 从Corner对象中提取2D坐标
        corners_2d = np.array([corner.point for corner in corners], dtype=np.float32)

        # 使用RANSAC求解PnP
        try:
            # 尝试使用返回4个值的版本（较新版本的OpenCV）
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=object_points.astype(np.float32),
                imagePoints=corners_2d,
                cameraMatrix=self._INTRINSIC_MAT,
                distCoeffs=self._DIST_COEFFS,
                reprojectionError=self._RANSAC_THRESH,
                flags=cv2.SOLVEPNP_EPNP,  # 使用EPNP算法
            )

            if not success or inliers is None or len(inliers) < 4:
                return None
        except TypeError:
            # 尝试使用返回3个值的版本（旧版本的OpenCV）
            rvec: cv2.typing.MatLike
            rvec, tvec, inliers = cv2.solvePnPRansac( # type: ignore
                object_points.astype(np.float32),
                corners_2d,
                self._INTRINSIC_MAT,
                self._DIST_COEFFS,
                flags=cv2.SOLVEPNP_EPNP,  # 使用EPNP算法
                reprojectionError=self._RANSAC_THRESH
            ) # type: ignore

            if inliers is None or len(inliers) < 4:
                return None

        return rvec, tvec

    def solve_multi_gate(self, gate_corners: List[Sequence[Corner]], gate_object_points: List[NDArray]) -> Optional[Tuple[NDArray, NDArray]]:
        """
        从多个门的角点求解PnP问题
        Args:
            gate_corners (List[Sequence[Corner]]): 多个门的角点序列列表
            gate_object_points (List[NDArray]): 多个门的3D世界点坐标列表
        Returns:
            Optional[Tuple[NDArray, NDArray]]: (旋转向量, 平移向量)，若失败则返回None
        """
        # 合并所有门的角点和对应的3D点
        all_corners = []
        all_object_points = []
        
        for corners, object_points in zip(gate_corners, gate_object_points):
            all_corners.extend(corners)
            all_object_points.append(object_points)
        
        if len(all_corners) < 4:
            return None
        
        # 拼接所有3D点
        merged_object_points = np.vstack(all_object_points)
        
        return self.solve(all_corners, merged_object_points)

    def get_pose(self, corners: Sequence[Corner], object_points: NDArray) -> Optional[Tuple[NDArray, NDArray]]:
        """
        获取相机位姿（旋转矩阵和平移向量）
        Args:
            corners (Sequence[Corner]): Corner类型对象组成的序列
            object_points (NDArray): 对应的3D世界点坐标 [N, 3]
        Returns:
            Optional[Tuple[NDArray, NDArray]]: (旋转矩阵, 平移向量)，若失败则返回None
        """
        result = self.solve(corners, object_points)
        if result is None:
            return None

        rvec, tvec = result
        # 将旋转向量转换为旋转矩阵
        rmat, _ = cv2.Rodrigues(rvec)
        return rmat, tvec

    def get_pose_quat(self, corners: Sequence[Corner], object_points: NDArray) -> Optional[Tuple[NDArray, NDArray]]:
        """
        获取相机位姿（四元数和平移向量）
        Args:
            corners (Sequence[Corner]): Corner类型对象组成的序列
            object_points (NDArray): 对应的3D世界点坐标 [N, 3]
        Returns:
            Optional[Tuple[NDArray, NDArray]]: (四元数, 平移向量)，若失败则返回None
        """
        result = self.get_pose(corners, object_points)
        if result is None:
            return None

        rmat, tvec = result
        # 将旋转矩阵转换为四元数
        quat = Rotation.from_matrix(rmat).as_quat()
        return quat, tvec

    def should_use_full_pnp(self, corners: Sequence[Corner], gate_distance: float) -> bool:
        """
        判断是否应该使用完整的PnP解决方案
        Args:
            corners (Sequence[Corner]): 角点序列
            gate_distance (float): 到门的距离（米）
        Returns:
            bool: 是否使用完整的PnP解决方案
        """
        return (len(corners) >= self._MIN_CORNERS and 
                self._MIN_DISTANCE <= gate_distance <= self._MAX_DISTANCE)

    # TODO: 移走EKF相关模块
    def get_fallback_pose(self, corners: Sequence[Corner], object_points: NDArray, ekf_attitude: NDArray) -> Optional[Tuple[NDArray, NDArray]]:
        """
        使用EKF姿态和PnP相对平移的备用姿态估计方法
        Args:
            corners (Sequence[Corner]): 角点序列
            object_points (NDArray): 对应的3D世界点坐标 [N, 3]
            ekf_attitude (NDArray): EKF估计的姿态四元数
        Returns:
            Optional[Tuple[NDArray, NDArray]]: (旋转矩阵, 平移向量)，若失败则返回None
        """
        # 求解PnP以获取相对平移
        result = self.solve(corners, object_points)
        if result is None:
            return None
        
        _, tvec = result
        
        # 使用EKF的姿态
        rmat = Rotation.from_quat(ekf_attitude).as_matrix()
        
        return rmat, tvec

    def is_attitude_update_reliable(self, num_gates: int, is_stationary: bool) -> bool:
        """
        判断PnP姿态更新是否可靠
        Args:
            num_gates (int): 检测到的门数量
            is_stationary (bool): 无人机是否静止
        Returns:
            bool: 姿态更新是否可靠
        """
        # 当检测到至少两个门时，或者无人机静止时，姿态更新是可靠的
        return num_gates >= 2 or is_stationary

    def reject_outlier(self, pnp_position: NDArray, ekf_position: NDArray, ekf_covariance: NDArray, num_corners: int) -> bool:
        """
        基于Kalman滤波器估计拒绝离群值
        Args:
            pnp_position (NDArray): PnP估计的位置
            ekf_position (NDArray): EKF估计的位置
            ekf_covariance (NDArray): EKF位置协方差矩阵
            num_corners (int): 检测到的角点数量
        Returns:
            bool: 是否接受PnP测量
        """
        # 计算位置差的平方范数
        position_diff = pnp_position - ekf_position
        position_diff_norm = np.linalg.norm(position_diff) ** 2
        
        # 计算阈值
        threshold = 16 * (num_corners ** 2) * np.trace(ekf_covariance)
        
        # 判断是否接受测量
        return position_diff_norm < threshold