from dataclasses import InitVar, dataclass, field
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
import yaml


@dataclass(frozen=True)
class DroneState:
    position: NDArray
    """无人机位置 [x, y, z]"""
    orientation: Rotation
    """无人机姿态（I2B）"""
    cam_position: NDArray
    """相机光心位置 [x, y, z]"""
    cam_orientation: Rotation
    """相机姿态（I2C）"""


@dataclass(frozen=True)
class GateExp:
    outer_half_w: float
    """门框外部半宽度（米）"""
    outer_half_h: float
    """门框外部半高度（米）"""
    inner_half_w: float
    """门框内部半宽度（米）"""
    inner_half_h: float
    """门框内部半高度（米）"""
    poses: List[List[float]]
    """所有门框坐标（m）和偏航（rad） [x, y, z, yaw]"""


@dataclass(frozen=True)
class CamExp:
    intrinsics: NDArray = field(init=False)
    """内参矩阵 (3,3)"""
    extrinsics: NDArray = field(init=False)
    """外参矩阵（相对于机体系）(3,4)"""
    dist_coeffs: NDArray = field(init=False)
    """畸变参数 (5,)"""
    img_w: int
    """图像宽度（像素）"""
    img_h: int
    """图像高度（像素）"""

    translation_B2C: NDArray = field(init=False)
    """相机相对于机体体系的平移向量 (3,)"""
    rotation_B2C: Rotation = field(init=False)
    """相机相对于机体体系的旋转"""

    _intrinsics_input: InitVar[List[List[float]]]
    _extrinsics_input: InitVar[List[List[float]]]
    _dist_coeffs_input: InitVar[List[float]]

    def __post_init__(
        self,
        _intrinsics_input: List[List[float]],
        _extrinsics_input: List[List[float]],
        _dist_coeffs_input: List[float],
    ):
        object.__setattr__(self, "intrinsics", np.array(_intrinsics_input))
        if self.intrinsics.shape != (3, 3):
            raise ValueError("intrinsics must be (3,3)")
        object.__setattr__(self, "extrinsics", np.array(_extrinsics_input))
        if self.extrinsics.shape != (3, 4):
            raise ValueError("extrinsics must be (3,4)")
        object.__setattr__(self, "dist_coeffs", np.array(_dist_coeffs_input))
        if self.dist_coeffs.shape != (5,):
            raise ValueError("dist_coeffs must be (5,)")
        object.__setattr__(self, "translation_B2C", self.extrinsics[:, 3])
        object.__setattr__(self, "rotation_B2C", Rotation.from_matrix(self.extrinsics[:, :3]))


@dataclass(frozen=True)
class Exp:
    gate: GateExp
    """门框设置"""
    cam: CamExp
    """相机设置"""

    @classmethod
    def load(cls, exp_name: str):
        """加载实验配置"""
        with open(f"exps/{exp_name}.yaml", "r") as f:
            config = yaml.safe_load(f)
        return cls(**config["gate"], **config["cam"])

    def create_drone_state(self, position: NDArray, orientation: Rotation):
        """创建无人机状态"""
        return DroneState(
            position=position,
            orientation=orientation,
            cam_position=np.array(self.cam.extrinsics[0]),
            cam_orientation=Rotation.from_matrix(self.cam.extrinsics[1]),
        )
