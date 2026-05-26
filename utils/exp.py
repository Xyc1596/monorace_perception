from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field, asdict, fields, is_dataclass
import os
from typing import Any, ClassVar, Dict, Generic, List, Tuple, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
import yaml


@dataclass(frozen=True)
class DroneState:
    position: NDArray = field(default_factory=lambda: np.array([0, 0, 0]))
    """无人机位置 [x, y, z]"""
    orientation: Rotation = field(default_factory=lambda: Rotation.from_euler("XYZ", [0, 0, 0]))
    """无人机姿态（I2B）"""
    cam_position: NDArray = field(default_factory=lambda: np.array([0, 0, 0]))
    """相机光心位置 [x, y, z]"""
    cam_orientation: Rotation = field(default_factory=lambda: Rotation.from_euler("XYZ", [0, 0, 0]))
    """相机姿态（I2C）"""

    cam_front: NDArray = field(default_factory=lambda: np.array([1, 0, 0]), init=False, repr=False)
    """相机前向单位向量 [x, y, z]"""

    def __post_init__(self) -> None:
        object.__setattr__(self, "cam_front", self.cam_orientation.apply(np.array([1, 0, 0])))


T = TypeVar("T", bound="IExp")


class IExp(ABC, Generic[T]):
    @classmethod
    def default(cls) -> T:
        return cast(T, cls(**{}))

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        pass


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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CamExp(IExp["CamExp"]):
    _IGNORED: ClassVar[List[str]] = ["translation_B2C", "rotation_B2C"]

    intrinsics_list: InitVar[List[List[float]]] = []
    """内参矩阵 (3,3)"""
    extrinsics_list: InitVar[List[List[float]]] = []
    """外参矩阵 (3,4)"""
    dist_coeffs_list: InitVar[List[float]] = []
    """畸变参数 (5,)"""

    intrinsics: NDArray = field(default_factory=lambda: np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), init=False)
    """内参矩阵 (3,3)"""
    extrinsics: NDArray = field(
        default_factory=lambda: np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]), init=False
    )
    """外参矩阵（相对于机体系）(3,4)"""
    dist_coeffs: NDArray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 0.0, 0.0]), init=False, repr=False)
    """畸变参数 (5,)"""
    img_w: int = field(default_factory=lambda: 640)
    """图像宽度（像素）"""
    img_h: int = field(default_factory=lambda: 480)
    """图像高度（像素）"""

    translation_B2C: NDArray = field(init=False)
    """相机相对于机体体系的平移向量 (3,)"""
    rotation_B2C: Rotation = field(init=False)
    """相机相对于机体体系的旋转"""

    def __post_init__(
        self,
        intrinsics_list: List[List[float]] = [],
        extrinsics_list: List[List[float]] = [],
        dist_coeffs_list: List[float] = [],
    ):
        if intrinsics_list:
            object.__setattr__(self, "intrinsics", np.array(intrinsics_list))
            if self.intrinsics.shape != (3, 3):
                raise ValueError("intrinsics must be (3,3)")
        if extrinsics_list:
            object.__setattr__(self, "extrinsics", np.array(extrinsics_list))
            if self.extrinsics.shape != (3, 4):
                raise ValueError("extrinsics must be (3,4)")
        if dist_coeffs_list:
            object.__setattr__(self, "dist_coeffs", np.array(dist_coeffs_list))
            if self.dist_coeffs.shape != (5,):
                raise ValueError("dist_coeffs must be (5,)")
        object.__setattr__(self, "translation_B2C", self.extrinsics[:, 3])
        object.__setattr__(self, "rotation_B2C", Rotation.from_matrix(self.extrinsics[:, :3]))

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in self._IGNORED}


@dataclass(frozen=True)
class ModuleExp(IExp["ModuleExp"]):
    crop_w: int = 384
    """自适应裁剪图像宽度（像素）"""
    crop_h: int = 384
    """自适应裁剪图像高度（像素）"""
    crop_gate_oblique_thresh: float = 1.0
    """自适应裁剪门框倾斜角阈值（rad），视野中倾斜角大于此值的门框会被排除"""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
        """加载实验配置"""
        with open(f"exps/{exp_name}.yaml", "r") as f:
            config = yaml.safe_load(f)
        return cls(**config["gate"], **config["cam"], **config["module"])

    def save(self, exp_name: str):
        """保存实验配置"""
        with open(f"exps/{exp_name}.yaml", "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f)

    def create_drone_state(self, position: NDArray, orientation: Rotation):
        """创建无人机状态"""
        return DroneState(
            position=position,
            orientation=orientation,
            cam_position=np.array(self.cam.extrinsics[0]),
            cam_orientation=Rotation.from_matrix(self.cam.extrinsics[1]),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate": self.gate.to_dict(),
            "cam": self.cam.to_dict(),
            "module": self.module.to_dict(),
        }


if __name__ == "__main__":
    exp = Exp.default()
    print(exp.to_dict())
