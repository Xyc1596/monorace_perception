from typing import Optional, Tuple

from filterpy.kalman import ExtendedKalmanFilter
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


class EKF:
    _gravity = np.array([[0], [0], [-9.81]])  # 重力加速度

    def __init__(
        self,
        imu_saturation_threshold: float = 22.0,
        acc_ema_alpha: float = 0.1,
    ):
        """
        初始化扩展卡尔曼滤波器
        状态向量 x : [位置(3), 速度(3), 四元数(4), 偏差(6)]

        Args:
            saturation_threshold (float, optional): IMU 加速度饱和阈值
                - IMU 测得加速度与模型预测加速度之差的欧几里得范数超过该阈值时，认为 IMU 饱和
            acc_ema_alpha (float, optional): 加速度指数移动平均平滑系数
        """
        # region 初始化EKF参数

        # 初始化EKF，状态维度16，测量维度7
        self._kf = ExtendedKalmanFilter(dim_x=16, dim_z=7)

        # 初始化状态向量
        # x = [
        #   pos_x, pos_y, pos_z,        # 0-2 位置
        #   vel_x, vel_y, vel_z,        # 3-5 速度
        #   q_w, q_x, q_y, q_z,         # 6-9 方向四元数
        #   bias_ax, bias_ay, bias_az,  # 10-12 加速度计偏差
        #   bias_gx, bias_gy, bias_gz   # 13-15 角速度陀螺偏差
        # ]
        self._kf.x = np.zeros(16)
        self._kf.x[6] = 1.0  # 四元数初始化为单位四元数

        # 初始化协方差矩阵
        self._kf.P = np.eye(16)
        self._kf.P[:3, :3] *= 1.0  # 位置不确定性
        self._kf.P[3:6, 3:6] *= 1.0  # 速度不确定性
        self._kf.P[6:10, 6:10] *= 0.1  # 四元数不确定性
        self._kf.P[10:, 10:] *= 0.01  # IMU 偏差不确定性

        # 过程噪声协方差
        self._kf.Q = np.eye(16)
        self._kf.Q[:3, :3] *= 0.01  # 位置噪声
        self._kf.Q[3:6, 3:6] *= 0.1  # 速度噪声
        self._kf.Q[6:10, 6:10] *= 0.01  # 四元数噪声
        self._kf.Q[10:, 10:] *= 0.001  # IMU 偏差噪声

        # 测量噪声协方差
        self._kf.R = np.eye(7)
        self._kf.R[:3, :3] *= 0.1  # 位置测量噪声
        self._kf.R[3:, 3:] *= 0.01  # 方向四元数测量噪声

        # endregion

        self._u: NDArray = np.zeros(6)
        """
        控制状态向量 (6,) [a_x, a_y, a_z, roll, pitch, yaw]
        - IMU 测量值
        - IMU饱和时使用模型加速度而非从加速度计测量的加速度
        """
        self._delta_t: float = 0.0
        """时间步长"""

        self._IMU_SATURATION_THRESHOLD = imu_saturation_threshold
        """IMU 加速度饱和阈值"""
        self._ACC_EMA_ALPHA = acc_ema_alpha
        """加速度指数移动平均平滑系数"""
        self._ema_imu_acc = np.zeros(3)
        """EMA 滤波后IMU加速度测量值"""
        self._ema_model_acc = np.zeros(3)
        """EMA 滤波后模型预测加速度"""

        # 模型预测加速度阈值
        self._MODEL_ACC_THRESHOLD = 2.0  # 模型预测与测量的差异阈值

    def _predict(self, imu_data: NDArray, dt: float, motor_commands: Optional[NDArray] = None):
        """
        预测步骤
        Args:
            imu_data: IMU数据 [ax, ay, az, gx, gy, gz]
            dt: 时间步长
            motor_commands: 电机命令（用于模型预测）
        """
        # 提取IMU数据
        acc_meas = imu_data[:3]  # 加速度 IMU 测量值
        gyro_meas = imu_data[3:]  # 角速度 IMU 测量值

        # 检查IMU是否饱和
        if np.max(np.abs(acc_meas)) > self._IMU_SATURATION_THRESHOLD:
            # 使用模型预测的加速度
            if motor_commands is not None:
                acc_pred = self._predict_acceleration(motor_commands)
                # 检查模型预测与测量的差异
                if np.linalg.norm(acc_meas - acc_pred) > self._MODEL_ACC_THRESHOLD:
                    acc_meas = acc_pred
                    # 增加不确定性
                    self._kf.P[3:6, 3:6] *= 2.0
                    self._kf.P[6:10, 6:10] *= 2.0

        # 提取当前状态
        pos = self._kf.x[:3]
        vel = self._kf.x[3:6]
        quat = self._kf.x[6:10]
        bias_acc = self._kf.x[10:13]
        bias_gyro = self._kf.x[13:]

        # 计算真实加速度和角速度
        acc_true = acc_meas - bias_acc
        gyro_true = gyro_meas - bias_gyro

        # 四元数转旋转矩阵
        rot = Rotation.from_quat(quat)
        R = rot.as_matrix()

        # 加速度转换到惯性系
        acc_inertial = R @ acc_true + np.array([0, 0, 9.81])  # 重力

        # 预测状态
        # 位置预测
        pos_pred = pos + vel * dt + 0.5 * acc_inertial * dt**2
        # 速度预测
        vel_pred = vel + acc_inertial * dt
        # 四元数预测（使用角速度）
        quat_pred = self._update_quaternion(quat, gyro_true, dt)
        # 偏置预测（假设偏置恒定）
        bias_acc_pred = bias_acc
        bias_gyro_pred = bias_gyro

        # 组合预测状态
        self._kf.x[:3] = pos_pred
        self._kf.x[3:6] = vel_pred
        self._kf.x[6:10] = quat_pred
        self._kf.x[10:13] = bias_acc_pred
        self._kf.x[13:] = bias_gyro_pred

        # 计算雅可比矩阵
        F = self._compute_jacobian(R, acc_true, gyro_true, dt)

        # 预测协方差
        self._kf.P = F @ self._kf.P @ F.T + self._kf.Q

    def predict(
        self,
    ):
        pass

    def update(self, pnp_data: NDArray, num_corners: int):
        """
        更新步骤
        Args:
            pnp_data: PnP测量数据 [x, y, z, q_w, q_x, q_y, q_z]
            num_corners: 检测到的角点数量
        Returns:
            bool: 是否接受测量
        """
        # 离群值拒绝
        if not self._reject_outlier(pnp_data[:3], num_corners):
            return False

        # 测量向量
        z = pnp_data

        # 观测函数
        hx = self._kf.x[:7]  # 位置和四元数

        # 计算观测雅可比矩阵
        H = np.zeros((7, 16))
        H[:3, :3] = np.eye(3)  # 位置观测
        H[3:, 6:10] = np.eye(4)  # 四元数观测

        # 执行更新
        self._kf.update(z, HJacobian=H, Hx=hx)

        return True

    def _filter_acceleration(self, imu_acc: NDArray, model_acc: NDArray) -> Tuple[bool, NDArray]:
        """获取EMA滤波后的加速度

        Args:
            imu_acc (NDArray): IMU加速度测量值 (3,)
            model_acc (NDArray): 四旋翼动力学模型预测加速度 (3,)

        Returns:
            Tuple[bool, NDArray]: (是否饱和, EMA滤波后的加速度 (3,))
        """
        self._ema_imu_acc = self._ACC_EMA_ALPHA * imu_acc + (1 - self._ACC_EMA_ALPHA) * self._ema_imu_acc
        self._ema_model_acc = self._ACC_EMA_ALPHA * model_acc + (1 - self._ACC_EMA_ALPHA) * self._ema_model_acc
        if np.linalg.norm(self._ema_imu_acc - self._ema_model_acc, ord=2) > self._MODEL_ACC_THRESHOLD:
            return True, self._ema_model_acc
        else:
            return False, self._ema_imu_acc

    def _predict_acceleration(self, motor_commands: NDArray) -> NDArray:
        """
        使用电机命令预测加速度
        Args:
            motor_commands: 电机命令 [m1, m2, m3, m4]
        Returns:
            NDArray: 预测的加速度 [ax, ay, az]
        """
        # 简化的四旋翼模型
        # 假设电机命令与 thrust 成正比
        total_thrust = np.sum(motor_commands)
        # 假设加速度与总推力成正比
        # 这里需要根据实际无人机参数调整
        acc_z = (total_thrust - 9.81) / 1.0  # 假设质量为1kg
        return np.array([0, 0, acc_z])

    def _update_quaternion(self, quat: NDArray, gyro: NDArray, dt: float) -> NDArray:
        """
        使用角速度更新四元数
        Args:
            quat: 当前四元数
            gyro: 角速度
            dt: 时间步长
        Returns:
            NDArray: 更新后的四元数
        """
        # 四元数更新
        rot = Rotation.from_quat(quat)
        rot = rot * Rotation.from_rotvec(gyro * dt)
        return rot.as_quat()

    def _compute_jacobian(self, R: NDArray, acc: NDArray, gyro: NDArray, dt: float) -> NDArray:
        """
        计算状态转移雅可比矩阵
        Args:
            R: 旋转矩阵
            acc: 加速度
            gyro: 角速度
            dt: 时间步长
        Returns:
            NDArray: 雅可比矩阵
        """
        F = np.eye(16)

        # 位置对速度的导数
        F[:3, 3:6] = np.eye(3) * dt

        # 速度对四元数的导数
        # 这里使用简化的近似
        F[3:6, 6:10] = self._compute_acc_quat_jacobian(R, acc)

        # 速度对加速度偏置的导数
        F[3:6, 10:13] = -R * dt

        # 四元数对陀螺仪偏置的导数
        F[6:10, 13:] = -self._compute_quat_gyro_jacobian(gyro, dt)

        return F

    def _compute_acc_quat_jacobian(self, R: NDArray, acc: NDArray) -> NDArray:
        """
        计算加速度对四元数的雅可比矩阵
        Args:
            R: 旋转矩阵
            acc: 加速度
        Returns:
            NDArray: 雅可比矩阵
        """
        # 简化实现
        return np.zeros((3, 4))

    def _compute_quat_gyro_jacobian(self, gyro: NDArray, dt: float) -> NDArray:
        """
        计算四元数对陀螺仪的雅可比矩阵
        Args:
            gyro: 角速度
            dt: 时间步长
        Returns:
            NDArray: 雅可比矩阵
        """
        # 简化实现
        return np.zeros((4, 3)) * dt

    def _reject_outlier(self, pnp_position: NDArray, num_corners: int) -> bool:
        """
        离群值拒绝
        Args:
            pnp_position: PnP估计的位置
            num_corners: 检测到的角点数量
        Returns:
            bool: 是否接受测量
        """
        # 获取当前位置估计和协方差
        ekf_position = self._kf.x[:3]
        ekf_covariance = self._kf.P[:3, :3]

        # 计算位置差的平方范数
        position_diff = pnp_position - ekf_position
        position_diff_norm = np.linalg.norm(position_diff) ** 2

        # 计算阈值
        threshold = 16 * (num_corners**2) * np.trace(ekf_covariance)

        # 判断是否接受测量
        return position_diff_norm < threshold

    def _compute_f(self, delta_t: float):
        v = self._kf.x[3:6]  # 速度
        q = self._kf.x[6:10]  # 方向四元数
        R = Rotation.from_quat(q).as_matrix()  # 机体坐标系到世界坐标系的旋转矩阵
        dot_p = v.T
        dot_v = R @ (self._u[0:3] - self._kf.x[10:13]).reshape(3, 1) + self._gravity
        dot_q = 

    @property
    def state(self) -> NDArray:
        """
        获取当前状态
        Returns:
            NDArray: 当前状态向量
        """
        return self._kf.x

    @property
    def position(self) -> NDArray:
        """
        获取当前位置
        Returns:
            NDArray: 当前位置 [x, y, z]
        """
        return self._kf.x[:3]

    @property
    def velocity(self) -> NDArray:
        """
        获取当前速度
        Returns:
            NDArray: 当前速度 [vx, vy, vz]
        """
        return self._kf.x[3:6]

    @property
    def attitude(self) -> NDArray:
        """
        获取当前姿态四元数
        Returns:
            NDArray: 当前姿态四元数 [qw, qx, qy, qz]
        """
        return self._kf.x[6:10]
