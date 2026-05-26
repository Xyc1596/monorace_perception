import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


from dataclasses import dataclass, field


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
