package main

import (
	"fmt"
	"log"
	"net"
	"strconv"
	"strings"
	"sync"
	"time"

	"marboris/training"
)

var (
	busy bool
	mu   sync.Mutex
)

// default port for server
const portDef = "8081"

// default PM for client
const (
	opOk    = "Ok"
	opFail  = "Failer"
	opIgnor = "Ignored"
)

// default val for training.
const (
	requireDef     = true
	rateDef        = 0.1
	hiddenNodesDef = 50
)

// شبیه سازی عملیات طولانی
func longOperation(rate float64, hiddenNodes int) error {
	fmt.Printf("Starting long operation with rate=%f and hiddenNodes=%d...\n", rate, hiddenNodes)
	training.CreateNeuralNetwork("en", rate, hiddenNodes)
	time.Sleep(1 * time.Second)
	fmt.Println("Operation completed.")
	// شبیه‌سازی یک خطای احتمالی برای ارسال پیام Failer
	// if time.Now().Second()%2 == 0 {
	//     return fmt.Errorf("simulated operation error")
	// }
	return nil
}

func handleRequest(conn net.Conn) {
	defer conn.Close()

	req := requireDef
	rate := rateDef
	hiddenNodes := hiddenNodesDef

	buf := make([]byte, 1024)
	n, err := conn.Read(buf)
	if err != nil {
		log.Println("Error reading from connection:", err)
		return
	}

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
				req = true
			}
		case "rate":
			rate, err = strconv.ParseFloat(value, 64)
			if err != nil {
				rate = 0.1
			}
		case "hiddensNodes":
			hiddenNodes, err = strconv.Atoi(value)
			if err != nil {
				hiddenNodes = hiddenNodesDef
			}
		}
	}

	fmt.Printf("background work: %v\n", req)

	// قفل برای جلوگیری از اجرای همزمان عملیات اصلی
	mu.Lock()
	if busy {
		mu.Unlock()
		// ارسال پیام Ignored اگر سرور مشغول باشد
		conn.Write([]byte(opIgnor))
		fmt.Println("Server is busy, ignoring request.")
		return
	}
	busy = true
	mu.Unlock()

	// اجرای عملیات اصلی
	var response string
	if req {
		// حالت بلوک‌شده: کلاینت منتظر می‌ماند
		err := longOperation(rate, hiddenNodes)
		if err != nil {
			response = opFail
		} else {
			response = opOk
		}
		conn.Write([]byte(response))
	} else {
		// حالت پس‌زمینه: عملیات اصلی انجام می‌شود اما کلاینت منتظر نمی‌ماند
		go func() {
			err := longOperation(rate, hiddenNodes)
			if err != nil {
				fmt.Println("Background operation failed")
			} else {
				fmt.Println("Background operation succeeded")
			}

			// بعد از اتمام عملیات، سرور دوباره آماده پردازش درخواست‌های جدید می‌شود
			mu.Lock()
			busy = false
			mu.Unlock()
		}()
		conn.Write([]byte(opOk))
	}

	// اگر حالت بلوک‌شده بود، قفل را پس از عملیات آزاد می‌کنیم
	if req {
		mu.Lock()
		busy = false
		mu.Unlock()
	}
}

func main() {
	listenPort := "0.0.0.0:" + portDef
	ln, err := net.Listen("tcp", listenPort) // استفاده از TCP
	if err != nil {
		log.Fatal(err)
	}
	defer ln.Close()

	fmt.Printf("Server is listening on port %s\n", portDef)

	for {
		conn, err := ln.Accept() // قبول اتصال از کلاینت
		if err != nil {
			log.Println("Error accepting connection:", err)
			continue
		}

		go handleRequest(conn)
	}
}
