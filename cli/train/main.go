package main

import (
    "fmt"
    "log"
    "net"
    "strconv"
    "strings"
    "sync"
    "time"
)

var busy bool
var mu sync.Mutex

// شبیه سازی عملیات طولانی
func longOperation(rate float64, hiddenNodes int) error {
    fmt.Printf("Starting long operation with rate=%f and hiddenNodes=%d...\n", rate, hiddenNodes)
    time.Sleep(10 * time.Second) // شبیه‌سازی عملیات طولانی
    fmt.Println("Operation completed.")
    // شبیه‌سازی یک خطای احتمالی برای ارسال پیام Failer
    // if time.Now().Second()%2 == 0 {
    //     return fmt.Errorf("simulated operation error")
    // }
    return nil
}

// تابع پردازش درخواست‌ها
func handleRequest(conn net.Conn) {
    defer conn.Close()

    // تنظیم مقادیر پیش‌فرض
    req := true
    rate := 0.1
    hiddenNodes := 50

    // دریافت پیام از کلاینت
    buf := make([]byte, 1024)
    n, err := conn.Read(buf)
    if err != nil {
        log.Println("Error reading from connection:", err)
        return
    }

    // پردازش پیام دریافتی
    message := string(buf[:n])
    fmt.Printf("Received: %s\n", message)

    // پارس کردن پارامترهای کلاینت
    params := strings.Split(message, ",")
    for _, param := range params {
        keyValue := strings.Split(param, "=")
        if len(keyValue) != 2 {
            continue
        }

        key := strings.TrimSpace(keyValue[0])
        value := strings.TrimSpace(keyValue[1])

        switch key {
        case "req":
            req, err = strconv.ParseBool(value)
            if err != nil {
                req = true // پیش‌فرض
            }
        case "rate":
            rate, err = strconv.ParseFloat(value, 64)
            if err != nil {
                rate = 0.1 // پیش‌فرض
            }
        case "hiddensNodes":
            hiddenNodes, err = strconv.Atoi(value)
            if err != nil {
                hiddenNodes = 50 // پیش‌فرض
            }
        }
    }

    // چاپ پارامترها
    fmt.Printf("req: %v, rate: %f, hiddensNodes: %d\n", req, rate, hiddenNodes)

    // اجرای عملیات اصلی بر اساس مقدار req
    if req {
        // حالت بلوک‌شده: کلاینت منتظر می‌ماند
        err := longOperation(rate, hiddenNodes)
        var response string
        if err != nil {
            response = "Failer"
        } else {
            response = "Ok"
        }
        conn.Write([]byte(response))
    } else {
        // حالت پس‌زمینه: کلاینت منتظر نمی‌ماند
        go func() {
            err := longOperation(rate, hiddenNodes)
            if err != nil {
                fmt.Println("Background operation failed")
            } else {
                fmt.Println("Background operation succeeded")
            }
        }()
        conn.Write([]byte("Ok"))
    }
}

func main() {
    ln, err := net.Listen("tcp", ":8081") // استفاده از TCP
    if err != nil {
        log.Fatal(err)
    }
    defer ln.Close()

    fmt.Println("Server is listening on port 8081")

    for {
        conn, err := ln.Accept() // قبول اتصال از کلاینت
        if err != nil {
            log.Println("Error accepting connection:", err)
            continue
        }

        go handleRequest(conn)
    }
}
