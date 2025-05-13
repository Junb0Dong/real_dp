import socket
import json
import time
from robot_receiver import RobotReceiver
from robot_controller import RobotController

if __name__ == "__main__":
    # 机器人 IP 和端口
    HOST = "10.15.127.226"  # 替换为实际机器人 IP
    # HOST = "127.0.0.1"  # 本地测试
    PORT = 9000

    # 创建发送类实例
    sender = RobotController(HOST, PORT)

    # 创建接收类实例，传递共享 socket
    receiver = RobotReceiver(sender.get_socket())

    

    try:
        # 发送目标位姿
        target_pose = [1.0, 2.0, 0.5, 0.0, 0.0, 1.57]  # [x, y, z, roll, pitch, yaw]
        # sender.send_target_pose(target_pose)
        sender.send_moveJ([0.5, -0.2, 0.6, -0.4, 1, -0.8], speed=1.05, acceleration=1.4, asynchronous=True)
        
        print("开始接收数据...")
        while True:
            # 接收响应
            response = receiver.receive_response()
            
            if not response:  # 如果没有接收到数据，可以选择退出
                print("未接收到数据，退出接收循环")
                break
            
            # 打印接收到的响应
            print(f"接收到的数据: {response}")
            
            # 可选：根据特定条件退出循环
            if response == "quit":  # 假设接收到 "quit" 表示结束
                print("接收到退出信号，停止接收数据")
                break
            
            time.sleep(0.1)  # 避免过度占用 CPU 资源

    except KeyboardInterrupt:
        print("用户中断程序")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        # 关闭连接
        sender.close()