"""
门框分割，输出分割掩码及图像
"""

from pathlib import Path
from typing import List, Union, Tuple, Optional

import numpy as np
from numpy.typing import NDArray
from torch import Tensor
from ultralytics import YOLO
from ultralytics.engine.results import Results

from utils.imports import cv2


class YOLOGateSeg:
    def __init__(
        self,
        model_path: Union[str, Path],
        brightness: int,
        conf_thresh: float,
        iou_thresh: float,
        max_det: int,
        img_size: Tuple[int, int],
    ):
        """
        YOLOv8 门框分割
        Args:
            model_path (Union[str, Path]): 模型路径
            brightness (int): 亮度调整
            conf_thresh (float): NMS 置信度阈值（置信度低于此阈值的结果会被丢弃）
            iou_thresh (float): NMS 交并比阈值（如果两个检测结果交并比大于此阈值，则置信度较低者会被丢弃）
            max_det (int): 每帧图片最大检测数量
            img_size (Tuple[int, int]): 输入图像尺寸（512x288）
        """
        self._model: YOLO = YOLO(model_path)

        self._BRIGHTNESS: int = brightness
        self._CONF_THRESH: float = conf_thresh
        self._IOU_THRESH: float = iou_thresh
        self._MAX_DET: int = max_det

        self._last_raw_img: Union[NDArray, Tensor] = np.zeros(img_size)
        self._last_correlated_img: Union[NDArray, Tensor] = self._last_raw_img
        self._last_seg_results: Optional[Results] = None

    def segment(self, img: cv2.typing.MatLike) -> Results:
        """
        运行
        Args:
            img (MatLike): 输入图像（512x288），cv2格式
        Returns:
            Results: 分割结果
        """
        self._last_raw_img = self._to_ndarray(img)
        # 亮度调整
        self._last_correlated_img = self._to_ndarray(
            img
            if self._BRIGHTNESS
            else cv2.add(img, np.ones(img.shape) * self._BRIGHTNESS)
        )
        # 运行结果
        result_list: List[Results] = self._model(
            source=img,
            conf=self._CONF_THRESH,
            iou=self._IOU_THRESH,
            max_det=self._MAX_DET,
            half=True,
            rect=False,
            verbose=False,
        )
        assert len(result_list) < 2
        self._last_seg_results = result_list[0]
        return self._last_seg_results

    @property
    def last_seg_results(self):
        assert self._last_seg_results is not None
        return self._last_seg_results

    def get_last_raw_img(self) -> NDArray:
        """原始输入图像（cv2，CPU）"""
        return self._to_ndarray(self._last_raw_img)

    def get_last_correlated_img(self) -> NDArray:
        """亮度调整后的输入图像（cv2，CPU）"""
        return self._to_ndarray(self._last_correlated_img)

    def create_last_seg_masks_img(self) -> NDArray[np.uint8]:
        """分割结果二值化掩码图像（cv2，CPU）(0~255)"""
        if self.last_seg_results.masks is None:
            return np.zeros(self._last_raw_img.shape, dtype=np.uint8)
        else:
            return (
                (self._to_ndarray(self.last_seg_results.masks.data) > 0.5) * 255
            ).astype(np.uint8)

    def create_last_seg_results_img(self) -> NDArray:
        """分割结果图像（cv2，CPU）"""
        if self.last_seg_results is None:
            return self.get_last_raw_img()
        else:
            return self.last_seg_results.plot()

    @staticmethod
    def _to_ndarray(arr: Union[NDArray, Tensor]) -> NDArray:
        return arr.cpu().numpy() if isinstance(arr, Tensor) else arr
