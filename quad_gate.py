from dataclasses import dataclass
from typing import List, Tuple, Optional, Any, Dict

from cv2 import cv2
import numpy as np
from torch import Tensor


# region ======== 辅助数据结构 ========

@dataclass
class LineSegment:
    """表示一条线段，包含两个端点坐标"""
    p1: np.ndarray  # shape (2,)
    p2: np.ndarray  # shape (2,)

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.p2 - self.p1))

    @property
    def direction(self) -> np.ndarray:
        """返回单位方向向量 (从p1指向p2)"""
        vec = self.p2 - self.p1
        norm = np.linalg.norm(vec)
        if norm < 1e-6:
            return np.array([0.0, 0.0])
        return vec / norm


@dataclass
class CornerCandidate:
    """角点候选，包含坐标和生成它的两条线段索引"""
    point: np.ndarray  # shape (2,)
    line_idx1: int
    line_idx2: int
    descriptor: Tuple[int, int, int, int]  # 四个方向上的二值像素值


# endregion

# region ======== 步骤1：去旋转掩码 ========

def derotate_mask(mask: np.ndarray, roll: float) -> np.ndarray:
    """
    根据无人机滚转角旋转二值掩码，使图像中的垂直轴与世界向上方向对齐。
    :param mask: 输入二值掩码，shape (H, W)，值0或1（或0/255）
    :param roll: 滚转角（弧度），正值表示无人机向右倾斜
    :return: 去旋转后的掩码，shape (H, W)，背景为0，前景为1（或255）
    """
    # 确保掩码为单通道二值图像（0/255格式便于OpenCV处理）
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    h, w = mask.shape
    center = (w / 2.0, h / 2.0)
    angle_deg = -np.degrees(roll)  # 抵消滚转，旋转方向相反
    rotation_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # 计算旋转后图像的新尺寸，确保完整内容
    cos = np.abs(rotation_mat[0, 0])
    sin = np.abs(rotation_mat[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rotation_mat[0, 2] += (new_w / 2) - center[0]
    rotation_mat[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(mask, rotation_mat, (new_w, new_h),
                             flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # 二值化（以防插值产生非0/255值）
    rotated = (rotated > 0).astype(np.uint8) * 255
    return rotated


# endregion

# region ======== 步骤2：LSD线段检测 ========

def detect_lines(mask: np.ndarray) -> List[LineSegment]:
    """
    使用OpenCV LSD算法检测二值掩码中的线段。
    :param mask: 二值图像（0/255）
    :return: 线段对象列表
    """
    # 参数设置：scale, sigma_scale, quant, ang_th
    lsd = cv2.createLineSegmentDetector(scale=0.8, sigma_scale=0.8,
                                        quant=25.0, ang_th=30.0)
    # lines 格式: N x 1 x 4 (x1,y1,x2,y2)
    lines, _, _, _ = lsd.detect(mask)
    if lines is None:
        return []
    segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        segments.append(LineSegment(p1=np.array([x1, y1]), p2=np.array([x2, y2])))
    return segments


# endregion

# region ======== 步骤3：扩展线段并计算交点 ========

def extend_segment(seg: LineSegment, factor: float = 5.0 / 3.0) -> LineSegment:
    """
    将线段两端均匀扩展，使总长度变为原来的 factor 倍。
    :param seg: 原始线段
    :param factor: 长度缩放因子（>1）
    :return: 扩展后的新线段
    """
    old_len = seg.length
    new_len = old_len * factor
    if old_len < 1e-6:
        return seg
    delta = (new_len - old_len) / 2.0
    direction = seg.direction
    new_p1 = seg.p1 - delta * direction
    new_p2 = seg.p2 + delta * direction
    return LineSegment(p1=new_p1, p2=new_p2)


def line_intersection(seg1: LineSegment, seg2: LineSegment) -> Optional[np.ndarray]:
    """
    计算两条无限直线的交点（不考虑线段范围）。
    返回交点坐标，若平行则返回None。
    """
    # 直线参数: p = p1 + t * d
    d1 = seg1.direction
    d2 = seg2.direction
    # 检查平行
    cross = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(cross) < 1e-6:
        return None
    # 求解交点
    delta = seg2.p1 - seg1.p1
    t1 = (delta[0] * d2[1] - delta[1] * d2[0]) / cross
    intersection = seg1.p1 + t1 * d1
    return intersection


def compute_corner_candidates(segments: List[LineSegment],
                              factor: float = 5.0 / 3.0) -> List[CornerCandidate]:
    """
    扩展所有线段，计算每对扩展线段之间的交点，生成角点候选。
    :param segments: 原始线段列表
    :param factor: 线段长度扩展因子
    :return: 角点候选列表（已去重，保留交点及其对应的线段索引）
    """
    if len(segments) < 2:
        return []
    # 扩展线段
    ext_segs = [extend_segment(seg, factor) for seg in segments]

    # 收集所有交点，用字典去重（四舍五入到整数像素）
    candidates_dict: Dict[Tuple[int, int], CornerCandidate] = {}
    n = len(ext_segs)
    for i in range(n):
        for j in range(i + 1, n):
            inter = line_intersection(ext_segs[i], ext_segs[j])
            if inter is None:
                continue
            x, y = int(round(inter[0])), int(round(inter[1]))
            # 简单去重：保留第一个遇到的交点
            if (x, y) not in candidates_dict:
                candidates_dict[(x, y)] = CornerCandidate(
                    point=np.array([float(x), float(y)], dtype=np.float32),
                    line_idx1=i,
                    line_idx2=j,
                    descriptor=(0, 0, 0, 0)  # 占位，后续填充
                )
    return list(candidates_dict.values())


# endregion

# region ======== 步骤4：提取手工描述符 ========

def extract_descriptor(mask: np.ndarray, candidate: CornerCandidate,
                       segments: List[LineSegment]) -> Tuple[int, int, int, int]:
    """
    从二值掩码中提取候选角点的描述符。
    描述符由沿两条相交线段的四个方向（正向/反向）移动5像素处的像素值构成。
    :param mask: 去旋转后的二值掩码（0/255）
    :param candidate: 角点候选
    :param segments: 原始线段列表（用于获取方向向量）
    :return: 四元组 (v1, v2, v3, v4)
    """
    # 获取两条线段的方向（单位向量）
    seg1 = segments[candidate.line_idx1]
    seg2 = segments[candidate.line_idx2]
    d1 = seg1.direction
    d2 = seg2.direction

    # 如果方向向量为零向量（线段退化），使用默认方向
    if np.linalg.norm(d1) < 1e-6:
        d1 = np.array([1.0, 0.0])
    if np.linalg.norm(d2) < 1e-6:
        d2 = np.array([0.0, 1.0])

    # 移动5像素
    step = 5.0
    offsets = [step * d1, -step * d1, step * d2, -step * d2]

    h, w = mask.shape
    descriptor = []
    for off in offsets:
        px = int(round(candidate.point[0] + off[0]))
        py = int(round(candidate.point[1] + off[1]))
        if 0 <= px < w and 0 <= py < h:
            val = mask[py, px]
            descriptor.append(1 if val > 0 else 0)
        else:
            descriptor.append(0)
    return tuple(descriptor)  # type: ignore


# endregion

# region ======== 步骤5：角点匹配与RANSAC仿射变换估计 ========

def match_corners_with_ransac(
        prior_corners: List[np.ndarray],  # 先验角点坐标列表，shape (4,2)
        prior_descriptors: List[Tuple[int, int, int, int]],
        candidates: List[CornerCandidate],
        mask: np.ndarray,
        segments: List[LineSegment],
        max_distance: float = 100.0,
        ransac_threshold: float = 5.0,
        max_translation: float = 150.0
) -> Optional[np.ndarray]:
    """
    将候选角点与先验角点进行匹配，并使用RANSAC估计仿射变换（4个自由度）。
    返回经过变换后的四个角点坐标（按照先验顺序），若失败则返回None。
    :param prior_corners: 先验角点坐标（在去旋转后的坐标系中）
    :param prior_descriptors: 每个先验角点的描述符
    :param candidates: 角点候选列表
    :param mask: 去旋转后的二值掩码
    :param segments: 原始线段列表（用于描述符计算）
    :param max_distance: 候选与对应先验的最大像素距离
    :param ransac_threshold: RANSAC内点阈值（像素）
    :param max_translation: 任何方向上的平移超过此值则拒绝解
    :return: 匹配后的四个角点坐标 (4,2)，或None
    """
    # 1. 为所有候选计算描述符
    for cand in candidates:
        cand.descriptor = extract_descriptor(mask, cand, segments)

    # 2. 根据描述符完全匹配 + 距离约束建立匹配对
    matches: List[Tuple[np.ndarray, np.ndarray]] = []  # (prior_pt, candidate_pt)
    for i, (prior_pt, prior_desc) in enumerate(zip(prior_corners, prior_descriptors)):
        best_match = None
        best_dist = float('inf')
        for cand in candidates:
            if cand.descriptor != prior_desc:
                continue
            dist = float(np.linalg.norm(cand.point - prior_pt))
            if dist <= max_distance and dist < best_dist:
                best_dist = dist
                best_match = cand.point
        if best_match is not None:
            matches.append((prior_pt, best_match))

    if len(matches) < 2:  # 至少需要两个点才能估计仿射变换
        return None

    # 3. 使用RANSAC估计仿射变换（平移 + 旋转 + 均匀缩放）
    src_pts = np.array([m[0] for m in matches], dtype=np.float32)
    dst_pts = np.array([m[1] for m in matches], dtype=np.float32)

    # estimateAffinePartial2D 要求至少2个点
    if len(src_pts) < 2:
        return None

    transform, inliers = cv2.estimateAffinePartial2D(
        src_pts, dst_pts, method=cv2.RANSAC, ransacThreshold=ransac_threshold
    )
    if transform is None or inliers is None or np.sum(inliers) < 2:
        return None

    # 4. 检查平移量
    tx, ty = transform[0, 2], transform[1, 2]
    if abs(tx) > max_translation or abs(ty) > max_translation:
        return None

    # 5. 将四个先验角点通过变换映射到当前图像
    prior_corners_arr = np.array(prior_corners, dtype=np.float32).reshape(-1, 2)
    # 添加齐次坐标 (x, y) -> (x, y, 1)
    ones = np.ones((prior_corners_arr.shape[0], 1), dtype=np.float32)
    pts_homo = np.hstack([prior_corners_arr, ones])  # (4,3)
    transformed = pts_homo @ transform.T  # (4,2)
    return transformed


# endregion

# region ======== 主处理函数 ========

def process_gate_frame(
        results: Any,  # YOLOv8 分割输出（假设可获取掩码）
        roll: float,  # 滚转角（弧度）
        prior_corners: List[Tuple[float, float]],  # 先验角点 (x,y)
        prior_descriptors: List[Tuple[int, int, int, int]],
        line_extend_factor: float = 5.0 / 3.0,
        max_match_distance: float = 100.0,
        ransac_threshold: float = 5.0,
        max_translation: float = 150.0
) -> List[np.ndarray]:
    """
    无人机前视图像门框检测与跟踪主函数。
    :param results: YOLOv8分割模型输出，需支持 results[0].masks.data 获取掩码张量
    :param roll: 状态估计的无人机滚转角（弧度）
    :param prior_corners: 上一帧或模板中的四个门框角点（图像坐标，单位像素）
    :param prior_descriptors: 对应每个角点的描述符（四元组）
    :param line_extend_factor: 线段长度扩展因子（默认5/3）
    :param max_match_distance: 候选角点与先验角点的最大匹配距离（像素）
    :param ransac_threshold: RANSAC内点阈值
    :param max_translation: 仿射变换平移分量上限
    :return: 当前帧中四个门框角点坐标 (4,2)，若失败则返回None

    @see https://docs.ultralytics.com/zh/tasks/segment/#val
    @see https://docs.ultralytics.com/zh/modes/predict/#masks

    TODO: QuAdGate检测内外共8个角点，但目前用的yolov8分割掩码没有抠出内门框，因此只有4个外角点
    """
    # 从YOLOv8结果中提取分割掩码
    # 假设 results 是 ultralytics.YOLO 的返回结果，掩码形状为 (N, H, W)
    output: List[np.ndarray] = []
    if not hasattr(results, 'masks') or results.masks is None:
        return output
    masks_data: Tensor = results.masks.data  # num_objects x H x W
    if masks_data.numel() == 0:
        return output

    # 取第一个检测目标的掩码，转换为numpy并二值化（0/255）
    masks_np: np.ndarray = (masks_data.cpu() if hasattr(masks_data, 'cpu') else masks_data).numpy()
    for mask_np in masks_np:
        # 二值化（0/255）
        mask: np.ndarray = (mask_np > 0.5).astype(np.uint8) * 255
        # 步骤1：去旋转掩码（仅使用滚转角）
        rotated_mask = derotate_mask(mask, roll)
        # 步骤2：LSD线段检测
        line_segments = detect_lines(rotated_mask)
        if len(line_segments) < 2:
            continue
        # 步骤3：计算角点候选
        candidates = compute_corner_candidates(line_segments, factor=line_extend_factor)
        if not candidates:
            continue
        # 步骤4+5：描述符提取与匹配
        final_corners = (match_corners_with_ransac(
            prior_corners=[np.array(pt, dtype=np.float32) for pt in prior_corners],
            prior_descriptors=prior_descriptors,
            candidates=candidates,
            mask=rotated_mask,
            segments=line_segments,
            max_distance=max_match_distance,
            ransac_threshold=ransac_threshold,
            max_translation=max_translation
        ))
        if final_corners is not None:
            output.append(final_corners)
    return output

# endregion

# region ======== 使用示例 ========

# if __name__ == '__main__':
#     from ultralytics import YOLO
#
#     model = YOLO("yolov8n-seg.pt")
#     results = model(frame)
#     prior_corners = [(100, 200), (300, 200), (300, 400), (100, 400)]  # 示例
#     prior_desc = [(1, 0, 1, 0), (0, 1, 0, 1), (1, 0, 0, 1), (0, 1, 1, 0)]  # 示例
#     corners = process_gate_frame(results, roll=0.1,
#                                  prior_corners=prior_corners, prior_descriptors=prior_desc)

# endregion
