import sys
import os
import time
import pathlib
import cv2
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from termcolor import cprint

# DP3 相关导入
from diffusion_policy_3d.workspace.base_workspace import BaseWorkspace
import diffusion_policy_3d.common.rotation_util as rotation_util

# UR5 相关导入
import pyrealsense2 as rs
from pymodbus.client import ModbusTcpClient
import threading
from typing import Dict

# System configs
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
os.environ['WANDB_SILENT'] = "True"
OmegaConf.register_new_resolver("eval", eval, replace=True)

class Logger:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'

    @staticmethod
    def _timestamp():
        return time.strftime("%Y-%m-%d %H:%M:%S")

    @classmethod
    def info(cls, msg):
        print(f"{cls.BLUE}[INFO]{cls.NC} {cls._timestamp()} | {msg}")
        sys.stdout.flush()

    @classmethod
    def success(cls, msg):
        print(f"{cls.GREEN}[OK]  {cls.NC} {cls._timestamp()} | {msg}")
        sys.stdout.flush()

    @classmethod
    def warn(cls, msg):
        print(f"{cls.YELLOW}[WARN]{cls.NC} {cls._timestamp()} | {msg}")
        sys.stdout.flush()

    @classmethod
    def error(cls, msg):
        print(f"{cls.RED}[ERR] {cls.NC} {cls._timestamp()} | {msg}")
        sys.stdout.flush()

log = Logger()

# [新增内容] 放在 RealSenseCamera 类的原位置或其下方
class L515PointCloudCamera:
    def __init__(self, serial="f1420840", width=640, height=480, fps=30, filter_magnitude=2, target_points=4096):
        self.serial = serial
        self.width = width
        self.height = height
        self.target_points = target_points
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        
        self.decimation = rs.decimation_filter()
        self.decimation.set_option(rs.option.filter_magnitude, filter_magnitude)
        
        # 平面过滤参数
        self.A, self.B, self.C, self.D = -0.02, 0.95, 0.32, -0.5
        self.TABLE_THRESHOLD = 0.005
        
        self.pc = rs.pointcloud()
        self.last_pc_cache = None 

        try:
            self.profile = self.pipeline.start(config)
            log.success(f"L515 PointCloud Camera Started: {serial}")
            log.info("Warming up L515...")
            time.sleep(3.0)
            for _ in range(20):
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=8000)
                    if frames.get_depth_frame():
                        break
                except:
                    pass
            log.success("L515 ready for point cloud collection.")
        except Exception as e:
            log.error(f"Failed to start L515: {e}")
            self.pipeline = None

    def read(self):
        if not self.pipeline:
            return None
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                if self.last_pc_cache is not None:
                    return self.last_pc_cache.copy()
                return None

            depth_frame = self.decimation.process(depth_frame)
            points = self.pc.calculate(depth_frame)
            vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)

            valid_mask = vtx[:, 2] != 0
            x_cond = (vtx[:, 0] > -0.6) & (vtx[:, 0] < 0.7)
            z_cond = (vtx[:, 2] < 1.4)
            pts_dist = (vtx[:, 0] * self.A + vtx[:, 1] * self.B + vtx[:, 2] * self.C + self.D)
            plane_cond = pts_dist < -self.TABLE_THRESHOLD
            
            final_mask = valid_mask & x_cond & z_cond & plane_cond
            cropped_pc = vtx[final_mask]

            num_cropped = len(cropped_pc)
            if num_cropped > 0:
                choice_idx = np.random.choice(num_cropped, self.target_points, replace=(num_cropped < self.target_points))
                final_pc = cropped_pc[choice_idx]
            else:
                final_pc = np.zeros((self.target_points, 3), dtype=np.float32)

            self.last_pc_cache = final_pc.copy()
            return final_pc

        except Exception as e:
            log.error(f"L515 read error: {e}")
            return self.last_pc_cache.copy() if self.last_pc_cache is not None else None

    def stop(self):
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass

class MyGripper:
    def __init__(self, ip_address='192.168.1.11', port=502, threshold=0.001, lock_duration=2.0):
        self.ip_address = ip_address
        self.port = port
        self.client = None
        self.lock = threading.Lock()
        
        # Trigger-and-lock control (solves unstable predictions)
        self.current_state = None  # Tracks: 'open' or 'closed'
        self.state_threshold = threshold  # Threshold on physical [0,1] values
        self.lock_duration = lock_duration  # Seconds to lock after triggering CLOSED
        self.lock_until_time = 0.0  # Time until which gripper is locked
        
        self.open_hw_position = 0    # Hardware: 0 = OPEN
        self.closed_hw_position = 255  # Hardware: 255 = CLOSED
        
        # Legacy tracking (kept for compatibility)
        self.last_target_pos = -1
        self.current_normalized_pos = 0.0

        self.connect()
        self.activate_gripper()
        
        # Log configuration
        log.info(f"Gripper trigger-and-lock control:")
        log.info(f"  Threshold: {self.state_threshold}")
        log.info(f"  Lock duration: {self.lock_duration}s after CLOSED trigger")
        log.info(f"  → Once value > {self.state_threshold}, close and lock for {self.lock_duration}s")

    def connect(self):
        try:
            if self.client:
                self.client.close()

            self.client = ModbusTcpClient(self.ip_address, port=self.port)

            if self.client.connect():
                log.success(f"Gripper connected to {self.ip_address}")
            else:
                log.error(f"Gripper connection failed to {self.ip_address}")

        except Exception as e:
            log.error(f"Gripper connection error: {e}")

    def activate_gripper(self):
        try:
            self.client.write_registers(0, [0x0100, 0x0000, 0x6464, 0, 0, 0, 0, 0])
            time.sleep(0.1)
        except Exception as e:
            log.warn(f"Gripper activation failed: {e}")

    def move(self, value: float):
        """Move gripper with trigger-and-lock control (solves unstable predictions).
        
        Args:
            value: Target position (physical convention, [0, 1] range)
                   0.0 = OPEN (ready to grasp)
                   1.0 = CLOSED (grasping)
        
        Trigger-and-Lock Strategy:
            1. Detect trigger: value > threshold → send CLOSED command
            2. Lock for N seconds: ignore all gripper commands during lock
            3. After lock expires: can respond to new commands
        
        This handles unstable predictions by:
        - Immediately responding to grasp signal (value > threshold)
        - Holding the grasp regardless of subsequent value changes
        - Only allowing open after sufficient time has passed
        
        Example (threshold=0.001, lock_duration=2.0s):
            t=0.0s: value=0.000 → OPEN (initial)
            t=1.0s: value=0.009 → CLOSED (triggered! lock until t=3.0s)
            t=1.1s: value=0.000 → ignored (locked)
            t=1.5s: value=0.080 → ignored (locked)
            t=2.0s: value=0.001 → ignored (locked)
            t=3.1s: value=0.000 → OPEN (lock expired, can respond)
        
        Hardware mapping:
            hardware=0 → gripper OPEN
            hardware=255 → gripper CLOSED
        """
        val_clamped = max(0.0, min(1.0, value))
        current_time = time.time()
        
        # Check if gripper is locked
        if current_time < self.lock_until_time:
            # Still locked, ignore command
            return  # Silently ignore (no log spam)
        
        # Not locked, can respond to commands
        
        # Initialize on first call
        if self.current_state is None:
            # Start in OPEN state
            self.current_state = 'open'
            hw_position = self.open_hw_position
            self.current_normalized_pos = 0.0
            self._send_cmd(hw_position)
            self.last_target_pos = hw_position
            log.info(f"Gripper initialized: OPEN")
            return
        
        # Determine desired state
        if val_clamped > self.state_threshold:
            desired_state = 'closed'
        else:
            desired_state = 'open'
        
        # Only act if state changes
        if desired_state != self.current_state:
            if desired_state == 'closed':
                # Trigger CLOSED
                hw_position = self.closed_hw_position
                self.current_normalized_pos = 1.0
                self._send_cmd(hw_position)
                self.current_state = 'closed'
                self.last_target_pos = hw_position
                
                # Lock for specified duration
                self.lock_until_time = current_time + self.lock_duration
                log.info(f"Gripper: OPEN -> CLOSED (value={val_clamped:.3f})")
                log.success(f"   Locked for {self.lock_duration}s (until {time.strftime('%H:%M:%S', time.localtime(self.lock_until_time))})")
            else:
                # CLOSED → OPEN (only happens after lock expires)
                hw_position = self.open_hw_position
                self.current_normalized_pos = 0.0
                self._send_cmd(hw_position)
                self.current_state = 'open'
                self.last_target_pos = hw_position
                log.info(f"Gripper: CLOSED -> OPEN (value={val_clamped:.3f})")
                log.success(f"   Command sent (hw={hw_position})")
        # else: same state, no action needed

    def _send_cmd(self, position_int):
        try:
            cmd = [0x0900, position_int, 0x6464, 0, 0, 0, 0, 1]
            self.client.write_registers(0, cmd)
        except Exception as e:
            log.warn(f"Gripper command failed: {e}, attempting reconnect...")
            try:
                self.connect()
                # Retry command after reconnect
                self.client.write_registers(0, cmd)
            except Exception as e2:
                log.error(f"Gripper command failed after reconnect: {e2}")

    def get_current_position(self) -> float:
        return self.current_normalized_pos * 255.0

class URRobot:
    def __init__(self, robot_ip: str = "192.168.1.2", no_gripper: bool = False, gripper_ip: str = "192.168.1.11", 
                 gripper_threshold: float = 0.001, gripper_lock_duration: float = 2.0):
        import rtde_control
        import rtde_receive

        try:
            self.robot = rtde_control.RTDEControlInterface(robot_ip)
            log.success(f"Connected to UR Robot at {robot_ip}")
        except Exception as e:
            log.error(f"UR Connection Error: {e}")
            raise e

        try:
            self.r_inter = rtde_receive.RTDEReceiveInterface(robot_ip)
        except Exception as e:
            log.error(f"UR Receive Interface Error: {e}")

        self._use_gripper = not no_gripper
        if self._use_gripper:
            try:
                self.gripper = MyGripper(ip_address=gripper_ip, threshold=gripper_threshold, 
                                        lock_duration=gripper_lock_duration)
                log.success(f"Gripper trigger-and-lock enabled (threshold={gripper_threshold}, lock={gripper_lock_duration}s)")
            except Exception as e:
                log.error(f"Gripper connection failed: {e}")
                self._use_gripper = False

        self._free_drive = False
        self.robot.endFreedriveMode()

    def num_dofs(self) -> int:
        return 7 if self._use_gripper else 6

    def get_joint_state(self) -> np.ndarray:
        """Get current joint state including gripper (closed-loop feedback).
        
        Returns:
            Joint positions [6 or 7] array:
            - [6]: robot joints only (no gripper)
            - [7]: robot joints + gripper position [0, 1] (0=OPEN, 1=CLOSED)
        """
        robot_joints = self.r_inter.getActualQ()
        if self._use_gripper:
            # Read actual gripper state (closed-loop feedback)
            # current_normalized_pos is in [0, 1] physical format
            gripper_pos = self.gripper.current_normalized_pos
            pos = np.append(robot_joints, gripper_pos)
        else:
            pos = robot_joints
        return np.array(pos)

    def get_observations(self) -> Dict[str, np.ndarray]:
        joints = self.get_joint_state()
        pos_quat = np.zeros(7)
        gripper_pos = np.array([joints[-1]]) if self._use_gripper else np.array([0.0])

        return {
            "joint_positions": joints,
            "joint_velocities": joints,
            "ee_pos_quat": pos_quat,
            "gripper_position": gripper_pos,
        }
    
class UR5EnvInference:
    """
    针对 UR5 + L515 点云的 DP3 推理环境封装
    """
    def __init__(self, obs_horizon=2, action_horizon=8, device="cuda", num_points=4096, frequency=10.0):
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.num_points = num_points
        self.dt = 1.0 / frequency
        self.device = torch.device(device)
        
        # 1. 硬件初始化
        cprint("Initializing L515 PointCloud Camera...", "cyan")
        self.camera_pc = L515PointCloudCamera(serial="f1420840", target_points=num_points)
        
        cprint("Initializing UR5 Robot...", "cyan")
        self.robot = URRobot(robot_ip="192.168.1.2", gripper_ip="192.168.1.11")
        self.home_joints = np.array([-1.57, -1.57, -1.57, -1.57, 1.57, 3.14])

        # 缓存 Buffer
        self.cloud_array = []
        self.agent_pos_array = []
        self.action_array = []
    
    def _get_robot_proprio(self):
        """获取真实的 10D 状态 (xyz, 6D rot, gripper)"""
        try:
            tcp_pose = self.robot.r_inter.getActualTCPPose()
            ee_xyz = np.array(tcp_pose[:3])  # meters
            ee_rot_vec = np.array(tcp_pose[3:6])  # rot vector

            # 转换为 6D 旋转
            rot_mat, _ = cv2.Rodrigues(ee_rot_vec)
            ee_rot_6d = np.concatenate([rot_mat[:, 0], rot_mat[:, 1]])

        except Exception as e:
            cprint(f"Failed to get TCP pose: {e}", "red")
            ee_xyz = np.zeros(3)
            ee_rot_6d = np.array([1, 0, 0, 0, 1, 0])

        gripper_state = self.robot.gripper.current_normalized_pos if self.robot._use_gripper else 0.0
        proprio = np.concatenate([ee_xyz, ee_rot_6d, [gripper_state]])
        
        return proprio

    def execute_action(self, action_10d):
        """执行 10D 动作指令"""
        try:
            target_pos = action_10d[:3]
            target_rot_6d = action_10d[3:9]
            target_gripper = action_10d[9]

            # 6D 还原为旋转矩阵，再转为旋转向量
            x = target_rot_6d[:3] / np.linalg.norm(target_rot_6d[:3])
            y = target_rot_6d[3:6] - np.dot(x, target_rot_6d[3:6]) * x
            y = y / np.linalg.norm(y)
            z = np.cross(x, y)
            target_rot_mat = np.column_stack([x, y, z])
            
            target_rot_vec, _ = cv2.Rodrigues(target_rot_mat)
            target_pose = np.concatenate([target_pos, target_rot_vec.flatten()])

            # 执行机械臂与夹爪
            self.robot.robot.moveL(target_pose.tolist(), 0.1, 0.3, False)
            if self.robot._use_gripper:
                self.robot.gripper.move(target_gripper)

        except Exception as e:
            cprint(f"Target proprio execution failed: {e}", "red")

    def _get_obs_dict(self):
        """打包 Observation 给模型"""
        agent_pos = np.stack(self.agent_pos_array[-self.obs_horizon:], axis=0)
        obs_cloud = np.stack(self.cloud_array[-self.obs_horizon:], axis=0)
            
        obs_dict = {
            'agent_pos': torch.from_numpy(agent_pos).float().unsqueeze(0).to(self.device),
            'point_cloud': torch.from_numpy(obs_cloud).float().unsqueeze(0).to(self.device)
        }
        return obs_dict

    def step(self, action_list):
        """执行动作序列并记录新状态"""
        for action_id in range(self.action_horizon):
            start_time = time.time()
            
            act = action_list[action_id]
            self.action_array.append(act)
            
            # 1. 下发物理指令
            self.execute_action(act)
            
            # 2. 读取新状态
            pc = self.camera_pc.read()
            if pc is None:
                pc = self.cloud_array[-1] # fallback
            
            proprio = self._get_robot_proprio()
            
            self.cloud_array.append(pc)
            self.agent_pos_array.append(proprio)
            
            # 控制频率
            elapsed = time.time() - start_time
            time.sleep(max(0, self.dt - elapsed))
            
        return self._get_obs_dict()
    
    def reset(self, confirm=True):
        """复位机器人并初始化 Buffer"""
        if confirm:
            input("\nMove robot to home position? Press Enter...")

        cprint("Resetting robot to home...", "yellow")
        current_joints = self.robot.get_joint_state()[:6]
        max_delta = np.abs(current_joints - self.home_joints).max()
        steps = min(int(max_delta / 0.01), 100)
        
        for joint_target in np.linspace(current_joints, self.home_joints, max(steps, 10)):
            self.robot.robot.moveJ(joint_target.tolist(), 0.3, 0.5, False)
            time.sleep(0.01)

        time.sleep(1.0) # 等待稳定
        cprint("Robot ready!", "green")
        
        # 清空 Buffer
        self.cloud_array, self.agent_pos_array, self.action_array = [], [], []

        # 获取初始观测，填充 obs_horizon
        pc = self.camera_pc.read()
        proprio = self._get_robot_proprio()

        for _ in range(self.obs_horizon):
            self.cloud_array.append(pc)
            self.agent_pos_array.append(proprio)
            
        return self._get_obs_dict()


@hydra.main(
    config_path=str(pathlib.Path(__file__).parent.joinpath('diffusion_policy_3d','config'))
)
def main(cfg: OmegaConf):
    torch.manual_seed(42)
    OmegaConf.resolve(cfg)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. 载入 Policy
    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    policy = workspace.get_model().to(device)
    policy.eval()
    
    # 获取预测长度（通常模型预测比如16步，我们执行前8步）
    action_horizon = policy.horizon - policy.n_obs_steps + 1

    # 2. 初始化环境
    env = UR5EnvInference(
        obs_horizon=policy.n_obs_steps, 
        action_horizon=action_horizon, 
        device=device,
        num_points=4096,
        frequency=10.0 # 根据你训练时的频率修改
    )

    obs_dict = env.reset(confirm=True)

    # 3. 部署参数
    roll_out_length = 300 
    step_count = 0
    record_data = True
    
    cprint("=== Starting DP3 Rollout ===", "green", attrs=["bold"])
    
    # 4. 主控循环
    while step_count < roll_out_length:
        with torch.no_grad():
            # 这里的输入 obs_dict (物理数值)，经过 policy 内部通常会自动 normalize
            # 预测出的 action 通常会自动 unnormalize 变回物理数值
            # 如果你的 DP3 实现不会自动 Normalization，则需要在这里手动添加归一化/反归一化代码
            action = policy(obs_dict)[0]
            action_list = [act.cpu().numpy() for act in action]
        
        # 执行动作序列
        obs_dict = env.step(action_list)
        step_count += action_horizon
        print(f"Executed Steps: {step_count}/{roll_out_length}")

    # 5. 保存遥操验证数据 (可选)
    if record_data:
        import h5py
        save_dir = "deploy_logs"
        os.makedirs(save_dir, exist_ok=True)
        
        record_file_name = f"{save_dir}/demo_{int(time.time())}.h5"
        with h5py.File(record_file_name, "w") as f:
            f.create_dataset("point_cloud", data=np.array(env.cloud_array))
            f.create_dataset("agent_pos", data=np.array(env.agent_pos_array))
            f.create_dataset("action", data=np.array(env.action_array))
        
        cprint(f"Data saved to {record_file_name}", "yellow")

if __name__ == "__main__":
    main()