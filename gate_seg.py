"""
门框分割，输出分割掩码及图像
"""
from pathlib import Path
from typing import List, Union, Tuple, Optional

import numpy as np
from cv2 import cv2
from numpy.typing import NDArray
from torch import Tensor
from ultralytics import YOLO
from ultralytics.engine.results import Results

cv2cuda = cv2.cuda if hasattr(cv2, 'cuda') else cv2


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
        :param model_path: 模型路径
        :param brightness: 亮度调整
        :param conf_thresh: NMS 置信度阈值（置信度低于此阈值的结果会被丢弃）
        :param iou_thresh: NMS 交并比阈值（如果两个检测结果交并比大于此阈值，则置信度较低者会被丢弃）
        :param max_det: 每帧图片最大检测数量
        :param img_size: 输入图像尺寸（512x288）
        """
        self.__model: YOLO = YOLO(model_path)

        self.__BRIGHTNESS: int = brightness
        self.__CONF_THRESH: float = conf_thresh
        self.__IOU_THRESH: float = iou_thresh
        self.__MAX_DET: int = max_det

        self.__last_raw_img: Union[NDArray, Tensor] = np.zeros(img_size)
        self.__last_correlated_img: Union[NDArray, Tensor] = self.__last_raw_img
        self.__last_seg_results: Optional[Results] = None

    def segment(self, img: Union[NDArray, Tensor]) -> Results:
        """
        运行
        :param img: 输入图像（512x288），cv2格式
        :return: 分割结果
        """
        self.__last_raw_img = self._to_ndarray(img)
        # 亮度调整
        img_ = img if self.__BRIGHTNESS == 0 else cv2cuda.add(img, self.__BRIGHTNESS)
        self.__last_correlated_img = self._to_ndarray(img_)
        # 运行结果
        result_list: List[Results] = self.__model(
            source=img_,
            conf=self.__CONF_THRESH,
            iou=self.__IOU_THRESH,
            max_det=self.__MAX_DET,
            half=True,
            rect=False,
            verbose=False
        )
        assert len(result_list) < 2
        self.__last_seg_results = result_list[0]
        return self.__last_seg_results

    @property
    def last_seg_results(self):
        assert self.__last_seg_results is not None
        return self.__last_seg_results

    def get_last_raw_img(self) -> NDArray:
        """原始输入图像（cv2，CPU）"""
        return self._to_ndarray(self.__last_raw_img)

    def get_last_correlated_img(self) -> NDArray:
        """亮度调整后的输入图像（cv2，CPU）"""
        return self._to_ndarray(self.__last_correlated_img)

    def create_last_seg_masks_img(self) -> NDArray[np.uint8]:
        """分割结果二值化掩码图像（cv2，CPU）(0~255)"""
        return (self._to_ndarray(self.last_seg_results.masks.data) > 0.5).astype(np.uint8) * np.uint8(255)

    def create_last_seg_results_img(self) -> NDArray:
        """分割结果图像（cv2，CPU）"""
        return self.__last_seg_results.plot()

    @staticmethod
    def _to_ndarray(arr: Union[NDArray, Tensor]) -> NDArray:
        return arr.cpu().numpy() if isinstance(arr, Tensor) else arr
