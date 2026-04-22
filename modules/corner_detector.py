"""
角点检测 / QuAdGate
"""

from typing import Tuple, List, Dict, Optional

import numpy as np
from utils.imports import cv2
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from utils import Corner, CornerFromMask, LineSegment


class CornerDetector:
    def __init__(self, intrinsic_mat: NDArray, extrinsic_mat: NDArray):
        self._INTRINSIC_MAT: NDArray = intrinsic_mat
        self._ROT_B2C = Rotation.from_matrix(extrinsic_mat[:3, :3])
        self._EXT_FACTOR: float = 5 / 3
        """线段长度缩放因子"""
        self._MATCH_DIST_THRESH: int = 100
        """如果先验角点与候选角点像素距离大于此值，则拒绝匹配"""
        self._RANSAC_THRESH: float = 5.0
        """RANSAC 内点阈值（像素）"""
        self._MAX_TRANSLATION: int = 150
        """RANSAC 任何方向上的平移超过此值则拒绝解"""

        self._lsd: cv2.LineSegmentDetector = cv2.createLineSegmentDetector(
            scale=0.8,
            sigma_scale=0.8,
            quant=25.0,
            ang_th=30.0,
        )

    def _derotate_mask(self, mask: NDArray[np.uint8], body_quat_est: NDArray) -> Tuple[NDArray[np.uint8], NDArray]:
        """
        掩码去旋转（使图像中的垂直轴与世界向上方向对齐） TODO: 去旋转是否考虑偏航和俯仰？旋转中心？
        Args:
            mask (NDArray[np.uint8]): 二值化掩码图像 [H,W]（0/255）
            body_quat_est (NDArray): 状态估计的无人机姿态四元数（I2B）
        Returns:
            Tuple[NDArray[np.uint8], NDArray]: 去旋转的掩码图像 [H,W]（0/255），去旋转矩阵 [2,3]
        """
        rot_i2c = self._ROT_B2C * Rotation.from_quat(body_quat_est)
        # 惯性系向上向量投影到图像平面
        vec_up = self._INTRINSIC_MAT @ rot_i2c.apply([0, 0, 1])
        if abs(vec_up[2]) < 1e-6:
            raise ValueError("世界向上向量投影到图像平面时深度为零，无法计算方向")  # TODO: 解决方案
        # 计算旋转角度
        project_rad = np.arctan2(vec_up[1] / vec_up[2], vec_up[0] / vec_up[2])  # 投影方向的角度
        target_rad = -np.pi / 2  # 图像垂直向上（v轴负向）
        rot_deg = np.degrees(target_rad - project_rad)
        # 旋转掩码图像
        h, w = mask.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, rot_deg, 1.0)
        return cv2.warpAffine(mask, rot_mat, (w, h), flags=cv2.INTER_NEAREST).astype(np.uint8), rot_mat

    def _detect_lines(self, mask: NDArray[np.uint8]) -> List[LineSegment]:
        """
        使用OpenCV LSD算法检测二值掩码中的线段。
        Args:
            mask (NDArray[np.uint8]): 去旋转后的二值图像（0/255）
        Returns:
            List[LineSegment]: 线段对象列表
        """
        # lines 格式: N x 1 x 4 (x1,y1,x2,y2)
        lines = self._lsd.detect(mask)[0]
        if lines is None:
            return []
        segments = []
        for line in lines:  # (x1, y1, x2, y2)
            segments.append(LineSegment.of_line(line[0], self._EXT_FACTOR))
        return segments

    @staticmethod
    def _compute_corner_candidates(
        extended_segments: List[LineSegment],
        mask: NDArray[np.uint8],
    ) -> List[CornerFromMask]:
        """
        计算每对扩展线段之间的交点，生成角点候选。
        Args:
            extended_segments (List[LineSegment]): 扩展线段列表
            mask (NDArray[np.uint8]): 去旋转后的二值图像（0/255）
        Returns:
            List[CornerFromMask]: 角点候选列表（已去重，保留交点及其对应的线段索引）
        """
        if (n := len(extended_segments)) < 2:
            return []
        # 收集所有交点，用字典去重（四舍五入到整数像素）
        candidates_dict: Dict[Tuple[int, int], CornerFromMask] = {}
        for i in range(n):
            for j in range(i + 1, n):
                inter = extended_segments[i].intersection(extended_segments[j])
                if inter is None:
                    continue
                point = int(round(inter[0])), int(round(inter[1]))
                # 简单去重：保留第一个遇到的交点
                if point not in candidates_dict:
                    candidates_dict[point] = CornerFromMask(
                        np.array(point),
                        extended_segments[i],
                        extended_segments[j],
                        mask,
                        5,
                    )
        return list(candidates_dict.values())

    def _match(
        self,
        prior_corners: List[Corner],
        candidate_corners: List[CornerFromMask],
    ) -> List[Tuple[Corner, CornerFromMask]]:
        """
        根据描述符完全匹配 + 距离约束建立角点匹配对
        Args:
            prior_corners (List[Corner]): 先验角点
            candidate_corners (List[CornerFromMask]): 候选角点
        Returns:
            List[Tuple[Corner, CornerFromMask]]: (prior, candidate)
        """
        matches: List[Tuple[Corner, CornerFromMask]] = []  # (prior, candidate)
        for prior_corner in prior_corners:
            best_match: Optional[CornerFromMask] = None
            best_dist: float = float("inf")
            for candidate_corner in candidate_corners:
                if not candidate_corner.descriptor_matched(prior_corner):
                    continue
                dist = candidate_corner.distance_to(prior_corner)
                if dist < self._MATCH_DIST_THRESH and dist < best_dist:
                    best_dist = dist
                    best_match = candidate_corner
            if best_match is not None:
                matches.append((prior_corner, best_match))
        return matches

    def _ransac_transform(
        self,
        prior_corners: List[Corner],
        matches: List[Tuple[Corner, CornerFromMask]],
    ) -> Optional[NDArray]:
        """
        使用RANSAC估计仿射变换（4个自由度）,对四个先验角点坐标进行变换
        Args:
            prior_corners (List[Corner]): 先验角点坐标（在去旋转后的坐标系中）
            matches (List[Tuple[Corner, CornerFromMask]]): 根据描述符完全匹配 + 距离约束建立的角点匹配对 (prior, candidate)
        Returns:
            Optional[NDArray]: 经过变换后的四个角点坐标（按照先验顺序），若失败则返回None
        """
        # estimateAffinePartial2D 要求至少2个点
        if len(matches) < 2:
            return None

        matched_prior_points = np.array([m[0].point for m in matches], dtype=np.float32)  # [N, 2]
        matched_candidate_points = np.array([m[1].point for m in matches], dtype=np.float32)  # [N, 2]
        transform, inliers = cv2.estimateAffinePartial2D(
            matched_prior_points,
            matched_candidate_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=self._RANSAC_THRESH,
        )
        if transform is None or inliers is None or np.sum(inliers) < 2:
            return None

        # 检查平移量
        tx: float = transform[0, 2]
        ty: float = transform[1, 2]
        if abs(tx) > self._MAX_TRANSLATION or abs(ty) > self._MAX_TRANSLATION:
            return None

        # 将四个先验角点通过变换映射到当前图像
        prior_corners_arr = np.array([c.point for c in prior_corners], dtype=np.float32).reshape(-1, 2)
        # 添加齐次坐标 (x, y) -> (x, y, 1)
        ones = np.ones((prior_corners_arr.shape[0], 1), dtype=np.float32)
        pts_homo = np.hstack([prior_corners_arr, ones])  # (4,3)
        transformed = pts_homo @ transform.T  # (4,2)
        return transformed

    def detect(self, mask: NDArray[np.uint8], body_quat_est: NDArray, prior_corners: List[Corner]) -> Optional[NDArray]:
        """
        QuAdGate
        Args:
            mask (NDArray[np.uint8]): 单个门框的二值化掩码图像 [H, W]（0/255）
            body_quat_est (NDArray):  状态估计的无人机姿态四元数（I2B）
            prior_corners (List[Corner]): 上一帧或模板中的四个门框角点
        Returns:
            NDArray: 当前帧中四个门框角点坐标 [4,2]，若失败则返回None
        """
        # 1. 掩码去旋转
        derotated_mask, derotate_mat = self._derotate_mask(mask, body_quat_est)
        # 2. LSD 线段检测
        line_segments = self._detect_lines(derotated_mask)
        if len(line_segments) < 2:
            return None
        # 3：计算角点候选
        candidate_corners = self._compute_corner_candidates(line_segments, derotated_mask)
        if not candidate_corners:
            return None
        # 4. 根据描述子 & 距离约束匹配先验角点和候选角点
        matches = self._match(prior_corners, candidate_corners)
        # 5. RANSAC 估计4 自由度仿射变换（平移、旋转、均匀缩放），剔除异常匹配、得到精确角点坐标
        transformed_prior_corners = self._ransac_transform(prior_corners, matches)  # (4,3)
        # 6. 将角点逆旋转回原图像
        if transformed_prior_corners is None:
            return None
        else:
            # 添加齐次坐标
            ones = np.ones((transformed_prior_corners.shape[0], 1), dtype=np.float32)
            return np.hstack([transformed_prior_corners, ones]) @ derotate_mat.T
