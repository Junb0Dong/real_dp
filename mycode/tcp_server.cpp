#include <iostream>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>

int main() {
    int server_fd, new_socket;           // 服务器套接字和客户端连接套接字
    struct sockaddr_in address;          // 服务器地址结构
    int addrlen = sizeof(address);       // 地址结构大小
    char buffer[1024] = {0};             // 数据缓冲区

    // 创建套接字
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("创建套接字失败");
        exit(EXIT_FAILURE);
    }

    // 设置服务器地址结构
    address.sin_family = AF_INET;         // IPv4 协议
    address.sin_addr.s_addr = INADDR_ANY; // 监听所有网络接口（包括 127.0.0.1）
    address.sin_port = htons(8080);       // 端口 8080，转换为网络字节序

    // 绑定套接字到 IP 和端口
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("绑定失败");
        exit(EXIT_FAILURE);
    }

    // 监听传入的连接
    if (listen(server_fd, 3) < 0) {
        perror("监听失败");
        exit(EXIT_FAILURE);
    }

    std::cout << "等待客户端连接..." << std::endl;

    // 接受客户端连接
    if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
        perror("接受连接失败");
        exit(EXIT_FAILURE);
    }

    std::cout << "连接已建立" << std::endl;

    // 接收数据
    int valread = recv(new_socket, buffer, 1024, 0);
    if (valread < 0) {
        perror("接收数据失败");
    } else {
        std::cout << "接收到的数据: " << buffer << std::endl;
    }

    // 关闭套接字
    close(new_socket);
    close(server_fd);

    return 0;
}