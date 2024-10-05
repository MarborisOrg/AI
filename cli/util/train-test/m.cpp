#include <iostream>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <chrono>
#include <thread>

int main() {
    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        std::cerr << "Error creating socket" << std::endl;
        return -1;
    }

    sockaddr_in server_addr;
    std::memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(8000);
    server_addr.sin_addr.s_addr = inet_addr("0.0.0.0");

    std::string message = "Hello UDP Server";

    while (true) {
        int send_result = sendto(sock, message.c_str(), message.size(), 0,
                                 (sockaddr*)&server_addr, sizeof(server_addr));
        if (send_result < 0) {
            std::cerr << "Error sending message" << std::endl;
            close(sock);
            return -1;
        }

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    close(sock);
    return 0;
}
