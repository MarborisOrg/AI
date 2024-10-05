package main

import (
    "fmt"
    "log"
    "net"
    "time"
)

func main() {
    addr, err := net.ResolveUDPAddr("udp", "0.0.0.0:8000")
    if err != nil {
        log.Fatal(err)
    }
    conn, err := net.DialUDP("udp", nil, addr)
    if err != nil {
        log.Fatal(err)
    }
    defer conn.Close()
    buf := make([]byte, 1024)
    for {
        _, err = conn.Write([]byte("Hello UDP Server"))
        if err != nil {
            log.Fatal(err)
        }
        n, err := conn.Read(buf)
        if err != nil {
            log.Fatal(err)
        }
        fmt.Println(string(buf[:n]))
        time.Sleep(time.Second)
    }
}
