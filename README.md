# MonoRace Perception

## 目录结构
### `assets`
测试用数据

### `modules`
模块列表

### `test`
单元测试代码

### `utils`
模块用到的工具类

## 输入输出
```mermaid
---
title: "各模块输入输出关系（暂定）"
---
flowchart TD
    AC["自适应裁剪\n（Adaptive Cropping）"]
    Seg["语义分割（GateNet）"]
    CD["角点提取（QuAdGate）"]
    PnP["位姿估计（PnP）"]
    EKF["状态模型（EKF）"]

    cam[/"相机"/]
    gates[/"门框定义"/]
    ctrl[/"控制"/]

    cam -->|"输入图像"| AC
    gates -->|"全体门框中心 & 角点3D坐标"| AC
    EKF -->|"相机位置 & 姿态预测"值| AC

    AC -->|"裁剪后图像（384x）"| Seg

    Seg -->|"`二值掩码（384x）`"| CD
    AC -->|"角点3D坐标"| CD
    EKF -->|"相机姿态预测值"| CD

    CD -->|"匹配的角点\n3D坐标 & 2D坐标"| PnP
    PnP -->|"无人机位置 & 姿态测量值"| EKF
    EKF -->|"无人机位置 & 姿态"| ctrl
```

> [!NOTE]
> * 部分参数（包括相机内外参）未列出，详见[`github.com/Xyc1596/monorace_perception/utils/exp.py`](utils/exp.py)（WIP）
> * 无人机和相机位姿等状态量预测和测量值均封装为`utils.exp.DroneState`类（WIP）

