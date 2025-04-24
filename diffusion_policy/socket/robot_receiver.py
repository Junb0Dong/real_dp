import socket
import json

class RobotReceiver:
    def __init__(self, socket: socket.socket):
        """
        初始化 RobotResponseReceiver，使用共享的 socket 接收响应。

        Args:
            socket (socket.socket): 从发送类传递的 socket 对象
        """
        self.socket = socket

    def receive_response(self):
        """
        接收并解析机器人响应。

        Returns:
            dict: 机器人返回的响应（JSON 格式解析后的字典）

        Raises:
            socket.error: 如果接收失败
            json.JSONDecodeError: 如果 JSON 解析失败
        """
        try:
            # 接收响应
            data = self.socket.recv(1024).decode('utf-8')
            response = json.loads(data)
            print(f"Received response: {response}")
            return response
        except socket.error as e:
            print(f"Receive error: {e}")
            raise
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            raise