import socket

def send_message(message, host='127.0.0.1', port=8080):
    try:
        # 创建一个 TCP 套接字
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # 连接到服务器
        client.connect((host, port))
        print(f"已连接到 {host}:{port}")
        
        # 发送消息
        client.send(message.encode())
        print(f"已发送消息: {message}")
        
        # 关闭连接
        client.close()
        print("连接已关闭")
    except ConnectionRefusedError:
        print(f"连接失败: 无法连接到 {host}:{port}，请确保服务器正在运行")
    except Exception as e:
        print(f"发生错误: {e}")

# 使用示例
if __name__ == "__main__":
    message = "Hello from Python!"
    send_message(message)