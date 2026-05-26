from abc import ABC
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Dict, Generic, List, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
import yaml

from utils.states import DroneState
from utils.yaml_utils import NestedList

T = TypeVar("T", bound="IExp")
S = TypeVar("S")


@dataclass(frozen=True)
class IExp(ABC, Generic[T]):
    @classmethod
    def default(cls) -> T:
        return cast(T, cls(**{}))

    def to_dict(self) -> Dict[str, Any]:
        _ignored = getattr(self.__class__, "_IGNORED", [])
        result = {}
        for f in fields(self):
            if f.name in _ignored:
                continue
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                result[f.name] = NestedList.of(value.tolist())
            elif isinstance(value, IExp):
                result[f.name] = value.to_dict()
            elif isinstance(value, list):
                result[f.name] = NestedList.of(value)
            else:
                result[f.name] = value
        return result


@dataclass(frozen=True)
class GateExp(IExp["GateExp"]):
    outer_half_w: float = 1.35
    """门框外部半宽度（米）"""
    outer_half_h: float = 1.35
    """门框外部半高度（米）"""
    inner_half_w: float = 0.7
    """门框内部半宽度（米）"""
    inner_half_h: float = 0.7
    """门框内部半高度（米）"""
    poses: List[List[float]] = field(default_factory=list)
    """所有门框坐标（m）和偏航（rad） [x, y, z, yaw]"""


@dataclass(frozen=True)
class CamExp(IExp["CamExp"]):
    _IGNORED: ClassVar[List[str]] = ["translation_B2C", "rotation_B2C"]

    intrinsics: NDArray[np.float_] = field(
        default_factory=lambda: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float_)
    )
    """内参矩阵 (3,3)"""
    extrinsics: NDArray[np.float_] = field(
        default_factory=lambda: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float_)
    )
    """外参矩阵（相对于机体系）(3,4)"""
    dist_coeffs: NDArray[np.float_] = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float_)
    )
    """畸变参数 (5,)"""
    img_w: int = field(default_factory=lambda: 640)
    """图像宽度（像素）"""
    img_h: int = field(default_factory=lambda: 480)
    """图像高度（像素）"""

    translation_B2C: NDArray = field(init=False)
    """相机相对于机体体系的平移向量 (3,)"""
    rotation_B2C: Rotation = field(init=False)
    """相机相对于机体体系的旋转"""

    def __post_init__(self):
        object.__setattr__(self, "translation_B2C", self.extrinsics[:, 3])
        object.__setattr__(self, "rotation_B2C", Rotation.from_matrix(self.extrinsics[:, :3]))


class ModuleCornerDetectorExp(IExp["ModuleCornerDetectorExp"]):
    line_extend_factor: float = 5 / 3
    """线段长度缩放因子"""
    match_distance_thresh: int = 100
    """如果先验角点与候选角点像素距离大于此值，则拒绝匹配"""
    ransac_thresh: float = 5.0
    """RANSAC内点阈值（像素）"""
    ransac_max_translation: int = 150
    """RANSAC 任何方向上的平移超过此值则拒绝解"""


@dataclass(frozen=True)
class ModuleExp(IExp["ModuleExp"]):
    ransac_thresh: float = 5.0
    """RANSAC内点阈值（像素）"""


@dataclass(frozen=True)
class Exp(IExp["Exp"]):
    gate: GateExp = field(default_factory=lambda: GateExp.default())
    """门框设置"""
    cam: CamExp = field(default_factory=lambda: CamExp.default())
    """相机设置"""
    module: ModuleExp = field(default_factory=lambda: ModuleExp.default())
    """模块参数"""

    @classmethod
    def load(cls, exp_name: str):
        """加载实验配置
        Args:
            exp_name (str): 实验配置文件名（不包含目录和扩展名）
        """
        with open(f"exps/{exp_name}.yaml", "r") as f:
            config = yaml.safe_load(f)
        return cls(
            gate=GateExp(**config["gate"]),
            cam=CamExp(**config["cam"]),
            module=ModuleExp(**config["module"]),
        )

    def save(self, exp_name: str):
        """保存实验配置
        Args:
            exp_name (str): 实验配置文件名（不包含目录和扩展名）
        """
        with open(f"exps/{exp_name}.yaml", "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, sort_keys=False)

    def create_drone_state(self, position: NDArray, orientation: Rotation):
        """创建无人机状态"""
        return DroneState(
            position=position,
            orientation=orientation,
            cam_position=np.array(self.cam.extrinsics[0]),
            cam_orientation=Rotation.from_matrix(self.cam.extrinsics[1]),
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "save":
            exp = Exp.default()
            exp.save("default")
        elif cmd == "load":
            exp = Exp.load("default")
            print(exp)
