#include <iostream>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <chrono>
#include <thread>


// Test Training Server
int main() {
    while (true) {
        // ایجاد سوکت TCP
        int sock = socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0) {
            std::cerr << "Error creating socket" << std::endl;
            return -1;
        }

        sockaddr_in server_addr;
        std::memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(8081);
        server_addr.sin_addr.s_addr = inet_addr("127.0.0.1");  // Or 0.0.0.0

        // اتصال به سرور
        if (connect(sock, (sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            std::cerr << "Error connecting to server" << std::endl;
            close(sock);
            return -1;
        }

        // ارسال پیام با پارامترهای req, rate و hiddensNodes
        std::string message = "req=false,rate=0.1,hiddensNodes=50";
        int send_result = send(sock, message.c_str(), message.size(), 0);
        if (send_result < 0) {
            std::cerr << "Error sending message" << std::endl;
            close(sock);
            return -1;
        }

        // دریافت پاسخ از سرور
        char buffer[1024];
        int recv_result = recv(sock, buffer, sizeof(buffer) - 1, 0);
        if (recv_result > 0) {
            buffer[recv_result] = '\0'; // اضافه کردن null terminator
            std::cout << "Server Response: " << buffer << std::endl;
        } else {
            std::cerr << "Error receiving message" << std::endl;
        }

        // بستن سوکت
        close(sock);

        // صبر برای ارسال پیام بعدی
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    return 0;
}
