"""
角点检测 / QuAdGate
"""

from dataclasses import dataclass
from typing import Generic, Tuple, List, Dict, Optional, TypeVar, cast

from loguru import logger
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from utils.corner import Corner3D
from utils.exp import Exp
from utils.imports import cv2
from utils import Corner2DFromMask, LineSegment
from utils.states import DroneState

T = TypeVar("T")
S = TypeVar("S")


@dataclass(frozen=True)
class Match(Generic[T, S]):
    matched_inds_first: NDArray[np.int_]
    matched_inds_second: NDArray[np.int_]
    matches: List[Tuple[T, S]]

    def __len__(self) -> int:
        return len(self.matches)

    def __bool__(self) -> bool:
        return len(self.matches) > 0

    @property
    def matched_lists(self) -> Tuple[List[T], List[S]]:
        first_list = []
        second_list = []
        for m in self.matches:
            first_list.append(m[0])
            second_list.append(m[1])
        return first_list, second_list


class CornerDetector:
    def __init__(self, exp: Exp):
        self._INTRINSIC_MAT: NDArray = exp.cam.intrinsics
        self._ROT_B2C = exp.cam.rotation_B2C
        self._EXT_FACTOR: float = 5 / 3 #TODO: use exp
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

        self._debug_img_bgr: Optional[cv2.typing.MatLike] = None
        self._debug_img_scale: float = 1.0

        logger.info("CornerDetector initialized")

    def _derotate_mask(self, mask: NDArray[np.uint8], cam_orientation: Rotation) -> Tuple[NDArray[np.uint8], NDArray]:
        """
        掩码去旋转（使图像中的垂直轴与世界向上方向对齐） TODO: 去旋转是否考虑偏航和俯仰？旋转中心？
        Args:
            mask (NDArray[np.uint8]): 二值化掩码图像 [H,W]（0/255）
            cam_orientation (Rotation): 相机姿态（I2C）
        Returns:
            Tuple[NDArray[np.uint8], NDArray]: 去旋转的掩码图像 [H,W]（0/255），去旋转矩阵 [2,3]
        """
        rot_i2c = self._ROT_B2C * cam_orientation
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
    def _compute_candidate_corners(
        extended_segments: List[LineSegment],
        mask: NDArray[np.uint8],
    ) -> List[Corner2DFromMask]:
        """
        计算每对扩展线段之间的交点，生成候选角点。
        Args:
            extended_segments (List[LineSegment]): 扩展线段列表
            mask (NDArray[np.uint8]): 去旋转后的二值图像（0/255）
        Returns:
            List[Corner2DFromMask]: 候选角点列表（已去重，保留交点及其对应的线段索引）
        """
        if (n := len(extended_segments)) < 2:
            return []
        # 收集所有交点，用字典去重（四舍五入到整数像素）
        candidates_dict: Dict[Tuple[int, int], Corner2DFromMask] = {}
        for i in range(n):
            for j in range(i + 1, n):
                inter = extended_segments[i].intersection(extended_segments[j])
                if inter is None:
                    continue
                point = int(round(inter[0])), int(round(inter[1]))
                # 简单去重：保留第一个遇到的交点
                if point not in candidates_dict:
                    candidates_dict[point] = Corner2DFromMask(
                        np.array(point),
                        extended_segments[i],
                        extended_segments[j],
                        mask,
                        5,
                    )
        return list(candidates_dict.values())

    def _match(
        self,
        prior_corners: List[Corner3D],
        candidate_corners: List[Corner2DFromMask],
    ) -> Match[Corner3D, Corner2DFromMask]:
        """
        根据描述符完全匹配 + 距离约束建立角点匹配对
        Args:
            prior_corners (List[Corner3D]): 先验角点
            candidate_corners (List[Corner2DFromMask]): 候选角点
        Returns:
            Tuple[NDArray[np.int_], NDArray[np.int_], List[Tuple[Corner3D, Corner2DFromMask]]]:
            * matched_inds_prior
            * matched_inds_candidate
            * (prior, candidate)
        """
        matched_inds_prior: List[int] = []
        matched_inds_candidate: List[int] = []
        matches: List[Tuple[Corner3D, Corner2DFromMask]] = []  # (prior, candidate)
        for prior_corner in prior_corners:
            best_match: Optional[Corner2DFromMask] = None
            best_dist: float = float("inf")
            for candidate_corner in candidate_corners:
                if not candidate_corner.descriptor_matched(prior_corner):
                    continue
                dist = candidate_corner.distance_to(prior_corner)
                if dist < self._MATCH_DIST_THRESH and dist < best_dist:
                    best_dist = dist
                    best_match = candidate_corner
            if best_match is not None:
                best_match.match_prior(prior_corner)
                matched_inds_prior.append(prior_corners.index(prior_corner))
                matched_inds_candidate.append(candidate_corners.index(best_match))
                matches.append((prior_corner, best_match))
        return Match(
            matched_inds_first=np.array(matched_inds_prior),
            matched_inds_second=np.array(matched_inds_candidate),
            matches=matches,
        )

    def _ransac_transform(
        self,
        match: Match[Corner3D, Corner2DFromMask],
    ) -> Optional[Match[Corner3D, Corner2DFromMask]]:
        """
        使用RANSAC估计仿射变换（4个自由度），剔除离群点 & 拒绝平移量过大的匹配结果
        Args:
            match (Match[Corner3D, Corner2DFromMask]): 角点匹配结果
        Returns:
            Optional[Match[Corner3D, Corner2DFromMask]]: 过滤后的角点匹配结果（索引为 match 中的索引）
        """
        # estimateAffinePartial2D 要求至少2个点
        if len(match) < 2:
            return None

        matched_prior_inds = match.matched_inds_first
        matched_candidate_inds = match.matched_inds_second

        matched_prior_points, matched_candidate_points = match.matched_lists
        matched_prior_points_arr = np.array(matched_prior_points)  # [N, 2]
        matched_candidate_points_arr = np.array(matched_candidate_points)  # [N, 2]
        transform, inliers = cv2.estimateAffinePartial2D(
            matched_prior_points_arr,
            matched_candidate_points_arr,
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

        inliers_mask = inliers.astype(bool)
        return Match(
            matched_inds_first=matched_prior_inds[inliers_mask],
            matched_inds_second=matched_candidate_inds[inliers_mask],
            matches=[match.matches[i] for i in inliers_mask],
        )

    def _display_debug_img(self):
        assert self._debug_img_bgr is not None
        resized = cv2.resize(
            self._debug_img_bgr,
            None,
            fx=self._debug_img_scale,
            fy=self._debug_img_scale,
            interpolation=cv2.INTER_NEAREST,
        )
        cv2.imshow("debug", resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def detect(
        self,
        mask: NDArray[np.uint8],
        drone_state: DroneState,
        prior_3d_corners: List[Corner3D],
        derotate_prior_corners: bool = True,
        debug_mode: bool = False,
    ) -> Optional[Tuple[NDArray, NDArray]]:
        """
        QuAdGate
        Args:
            mask (NDArray[np.uint8]): 二值化掩码图像 [H, W]（0/255）
            drone_state (DroneState): 无人机状态估计
            prior_3d_corners (List[Corner3D]): 先验门框角点
            derotate_prior_corners (bool, optional): 是否对先验角点进行去旋转变换
            debug_mode (bool, optional): 是否开启调试模式，用于可视化
        Returns:
            Tuple[NDArray, NDArray]: 角点匹配对 (N, 3) & (N, 2)，若失败则返回None
        """
        # 1. 掩码去旋转
        derotated_mask, derotate_mat = self._derotate_mask(mask, drone_state.cam_orientation)
        if derotate_prior_corners:
            prior_corners_arr = np.array([c.point for c in prior_3d_corners]).reshape(-1, 1, 2)
            derotated_prior_3d_corners_arr = cv2.transform(prior_corners_arr, derotate_mat).reshape(-1, 2)
            derotated_prior_3d_corners = [
                Corner3D(
                    derotated_prior_3d_corners_arr[i],
                    prior_3d_corners[i].position,
                    prior_3d_corners[i].descriptor,
                    prior_3d_corners[i].gate_id,
                )
                for i in range(derotated_prior_3d_corners_arr.shape[0])
            ]
        else:
            derotated_prior_3d_corners = prior_3d_corners

        # 2. LSD 线段检测
        line_segments = self._detect_lines(derotated_mask)
        # if debug_mode:
        #     self._debug_img_bgr = cv2.cvtColor(derotated_mask, cv2.COLOR_GRAY2BGR)
        #     self._debug_img_scale = max(1, 512 // self._debug_img_bgr.shape[0])  # 小图先放大再显示
        #     for line in line_segments:
        #         line.plot(self._debug_img_bgr)  # 灰色线
        #     for prior_corner in prior_corners:
        #         prior_corner.plot(self._debug_img_bgr)  # 红色：先验角点
        if len(line_segments) < 2:
            logger.warning("No line segments detected")
            # if debug_mode:
            #     self._display_debug_img()
            return None

        # 3：计算角点候选
        derotated_candidate_corners = self._compute_candidate_corners(line_segments, derotated_mask)
        # if debug_mode:
        #     for corner in candidate_corners:
        #         corner.plot(self._debug_img_bgr)  # type: ignore

        if not derotated_candidate_corners:
            logger.warning("No candidate corners detected")
            # if debug_mode:
            #     self._display_debug_img()
            return None

        # 4. 根据描述子 & 距离约束匹配先验角点和候选角点
        match = self._match(derotated_prior_3d_corners, derotated_candidate_corners)
        if not match:
            logger.warning("No matches found")
            # if debug_mode:
            #     self._display_debug_img()
            return None

        # 5. RANSAC 估计 4 自由度仿射变换（平移、旋转、均匀缩放），剔除异常匹配 & 拒绝平移量过大的匹配结果
        ransac_match = self._ransac_transform(match)  # (4,3)
        if not ransac_match:
            logger.warning("RANSAC failed")
            return None

        # 6. 候选角点去旋转逆变换
        derotated_candidate_corners_filtered = [
            derotated_candidate_corners[cast(int, i)] for i in ransac_match.matched_inds_second
        ]
        derotated_candidate_points_filtered_arr = np.array([c.point for c in derotated_candidate_corners_filtered])
        return (
            np.array([prior_3d_corners[cast(int, i)].position for i in ransac_match.matched_inds_first]),
            cv2.transform(derotated_candidate_points_filtered_arr, cv2.invertAffineTransform(derotate_mat)),
        )
