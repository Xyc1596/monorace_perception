from typing import TypeVar, List, cast

import numpy as np
import yaml

T = TypeVar("T")


class NestedList(List[T]):
    """附带嵌套深度信息的列表类"""

    def __init__(self, *args: T, nest_level: int):
        """
        初始化 NestedList 实例。
        Args:
            args (*T): 列表中的元素
            nest_level (int): 当前列表的嵌套深度（最内层为 0）
        """
        super().__init__(args)
        self._nest_level = nest_level

    @property
    def nest_level(self) -> int:
        """返回当前列表的嵌套深度（最内层为 0）"""
        return self._nest_level

    @classmethod
    def of(cls, raw_list: List[T]) -> "NestedList[T]":
        """
        从普通嵌套列表构建 NestedList，自动计算各层深度。
        Args:
            raw_list (List[T]): 可能嵌套的普通列表
        Returns:
            NestedList[T]: 转换后的 NestedList 实例，最外层深度为最大嵌套深度
        """
        # 收集转换后的元素并计算最大子深度
        converted = []
        max_child_nest = -1
        for elem in raw_list:
            if isinstance(elem, list):
                sub = cls.of(elem)
                converted.append(sub)
                max_child_nest = max(max_child_nest, sub.nest_level)
            else:
                converted.append(elem)

        # 当前列表的深度 = 最大子深度 + 1（若无子列表则为 0）
        nest = max_child_nest + 1 if max_child_nest != -1 else 0
        return cls(*converted, nest_level=nest)

    def __repr__(self) -> str:
        """打印 NestedList 及其深度信息。"""
        inner = super().__repr__()
        return f"NestedList({inner}, nest_level={self._nest_level})"


yaml.add_representer(
    NestedList,
    lambda dumper, data: dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=data.nest_level == 0),
)

yaml.add_constructor(
    "tag:yaml.org,2002:seq",
    lambda loader, node: np.array(loader.construct_sequence(cast(yaml.SequenceNode, node), True)),
    Loader=yaml.SafeLoader,
)
