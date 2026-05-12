import sys, os

sys.path.append(".")
print(os.path.abspath(sys.path[-1]))
import numpy as np
from numpy.typing import NDArray
from utils.corner import Corner
from utils.imports import cv2
from utils.corner_descriptor import CornerDescriptor
from scipy.spatial.transform import Rotation
from modules.corner_detector import CornerDetector

# 内参矩阵
cam_S = 36  # 图像传感器水平 & 垂直尺寸 (mm)
f = 75  # 焦距 (mm)
h = 128
w = 128
intrinsic_mat = np.array(
    [
        [f * w / cam_S, 0, w / 2],
        [0, f * h / cam_S, h / 2],
        [0, 0, 1],
    ]
)

# 外参矩阵
extrinsic_mat = np.array(
    [
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ]
)

# 图像
img = cv2.imread("assets/binary_test-1.jpg", cv2.IMREAD_GRAYSCALE)
_, img_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
img_binary: NDArray

body_quat = Rotation.from_euler("zyx", [-30, 20, 0], degrees=True).as_quat()

prior_corners = [
    Corner(np.array([33, 25]), CornerDescriptor.LT),
    Corner(np.array([89, 33]), CornerDescriptor.RT),
    Corner(np.array([86, 93]), CornerDescriptor.RB),
    Corner(np.array([35, 80]), CornerDescriptor.LB),
]

detector = CornerDetector(intrinsic_mat, extrinsic_mat)
detected_corners = detector.detect(img_binary, body_quat, prior_corners, derotate_prior_corners=True, debug_mode=True)
print(detected_corners)
