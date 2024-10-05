package main

import (
    "fmt"
    "log"
    "net"
    "sync"
    "time"
)

var busy bool
var mu sync.Mutex

func longOperation() {
    fmt.Println("Starting long operation...")
    time.Sleep(10 * time.Second) // شبیه‌سازی عملیات طولانی
    fmt.Println("Operation completed.")
}

func handleRequest(ln *net.UDPConn, addr *net.UDPAddr, buf []byte, n int) {
    mu.Lock()
    if busy {
        mu.Unlock()
        fmt.Printf("Ignoring request from %s because the server is busy\n", addr)
        return
    }
    busy = true
    mu.Unlock()

    fmt.Printf("Received %s from %s\n", string(buf[:n]), addr)

    // شبیه‌سازی عملیات طولانی
    longOperation()

    // ارسال پاسخ
    _, err := ln.WriteToUDP([]byte("Operation completed: "+time.Now().String()), addr)
    if err != nil {
        log.Fatal(err)
    }

    mu.Lock()
    busy = false
    mu.Unlock()
}

func main() {
    addr, err := net.ResolveUDPAddr("udp", ":8000")
    if err != nil {
        log.Fatal(err)
    }
    ln, err := net.ListenUDP("udp", addr)
    if err != nil {
        log.Fatal(err)
    }
    defer ln.Close()

    fmt.Println("Listening on port 8000")
    buf := make([]byte, 1024)

    for {
        n, addr, err := ln.ReadFromUDP(buf)
        if err != nil {
            log.Fatal(err)
        }

        go handleRequest(ln, addr, buf, n)
    }
}
