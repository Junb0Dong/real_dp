import socket
import json

class RobotController:
    def __init__(self, host: str, port: int):
        """
        初始化 RobotCommandSender，创建并连接到机器人的 socket。

        Args:
            host (str): 机器人 IP 地址
            port (int): 机器人端口号
        """
        self.host = host
        self.port = port
        self.socket = None
        self.connect()

    def connect(self):
        """
        创建 socket 并连接到机器人。
        如果连接失败，抛出异常。
        """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self.socket.connect((self.host, self.port))
            print(f"Connected to robot at {self.host}:{self.port}")
        except socket.error as e:
            print(f"Connection failed: {e}")
            raise

    def send_target_pose(self, pose: list):
        """
        发送目标位姿到机器人。

        Args:
            pose (list): 目标位姿，例如 [x, y, z, roll, pitch, yaw]

        Raises:
            socket.error: 如果发送失败
        """
        try:
            # 构造命令（JSON 格式）
            command = {
                "command": "set_target_pose",
                "pose": pose
            }
            # 转换为 JSON 字符串并编码
            command_str = json.dumps(command)
            self.socket.sendall(command_str.encode('utf-8'))
            print(f"Sent target pose: {pose}")
        except socket.error as e:
            print(f"Send error: {e}")
            raise
    
    def send_moveJ(self, q: list, speed: float = 1.00, acceleration: float = 1.00, asynchronous: bool = False):
        """
        发送 moveJ 命令到机器人，控制机械臂移动到指定关节位置。

        Args:
            q (list): 关节位置，6 个浮点数列表 [q1, q2, q3, q4, q5, q6]，单位：弧度
            speed (float): 关节速度 [rad/s]，默认 1.05
            acceleration (float): 关节加速度 [rad/s^2]，默认 1.4
            asynchronous (bool): 是否异步执行，默认 False（阻塞直到完成）

        Raises:
            ValueError: 如果 q 长度不为 6
            socket.error: 如果发送失败
        """
        if len(q) != 6:
            raise ValueError("Joint positions must contain exactly 6 values for a 6-DOF arm")

        try:
            # 构造 moveJ 命令（JSON 格式）
            command = {
                "command": "moveJ",
                "q": q,
                "speed": speed,
                "acceleration": acceleration,
                "asynchronous": asynchronous
            }
            # 转换为 JSON 字符串并编码
            command_str = json.dumps(command)
            self.socket.sendall(command_str.encode('utf-8'))
            print(f"Sent moveJ command: q={q}, speed={speed}, acceleration={acceleration}, asynchronous={asynchronous}")
        except socket.error as e:
            print(f"Send error: {e}")
            raise

    def get_socket(self):
        """
        返回 socket 对象，供接收类使用。

        Returns:
            socket: 当前的 socket 对象
        """
        return self.socket

    def close(self):
        """
        关闭 socket 连接。
        """
        if self.socket:
            self.socket.close()
            print("Socket closed")
            self.socket = None