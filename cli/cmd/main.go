package main

import (
	"bufio"
	"context"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"strings"
	"sync"

	"github.com/gobwas/ws"
	"github.com/gobwas/ws/wsutil"
)

const (
	logFileName    = "logfile.log"
	configFileName = "config.json"
	defaultPort    = "8080"
	botName        = "marboris"
	hostName       = "localhost"
	defaultSsl     = false
)

type Client struct {
	Info      map[string]interface{}
	Locale    string
	Token     string
	Conn      net.Conn
	mu        sync.Mutex
}

type RequestMessage struct {
	Type        int                    `json:"type"`
	Content     string                 `json:"content"`
	Token       string                 `json:"user_token"`
	Information map[string]interface{} `json:"information"`
	Locale      string                 `json:"locale"`
}

type ResponseMessage struct {
	Content     string                 `json:"content"`
	Information map[string]interface{} `json:"information"`
}

type Config struct {
	Port    string `json:"port"`
	Host    string `json:"host"`
	SSL     bool   `json:"ssl"`
	BotName string `json:"bot_name"`
}

func NewClient(host string, ssl bool) (*Client, error) {
	scheme := "ws"
	if ssl {
		scheme += "s"
	}
	url := fmt.Sprintf("%s://%s/websocket", scheme, host)

	// ایجاد context
	ctx := context.Background()

	// ایجاد اتصال WebSocket با استفاده از context و gobwas/ws
	conn, _, _, err := ws.Dialer{}.Dial(ctx, url)
	if err != nil {
		logError("WebSocket connection failed", err)
		return nil, err
	}
	client := &Client{
		Info:   make(map[string]interface{}),
		Locale: "en",
		Token:  generateToken(),
		Conn:   conn,
	}
	if err := client.handshake(); err != nil {
		logError("Handshake failed", err)
		return nil, err
	}
	return client, nil
}

func (c *Client) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.Conn.Close()
}

func (c *Client) SendMessage(content string) (ResponseMessage, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	msg := RequestMessage{
		Type:        1,
		Content:     content,
		Token:       c.Token,
		Information: c.Info,
		Locale:      c.Locale,
	}
	data, err := json.Marshal(msg)
	if err != nil {
		logError("Marshaling message failed", err)
		return ResponseMessage{}, err
	}

	// ارسال پیام با استفاده از wsutil
	if err := wsutil.WriteClientText(c.Conn, data); err != nil {
		logError("Sending message failed", err)
		return ResponseMessage{}, err
	}

	// دریافت پاسخ
	respData, err := wsutil.ReadServerText(c.Conn)
	if err != nil {
		logError("Reading response failed", err)
		return ResponseMessage{}, err
	}

	var resp ResponseMessage
	if err := json.Unmarshal(respData, &resp); err != nil {
		logError("Unmarshaling response failed", err)
		return ResponseMessage{}, err
	}

	return resp, nil
}

func (c *Client) handshake() error {
	msg := RequestMessage{
		Type:        0,
		Information: c.Info,
		Token:       c.Token,
	}
	data, err := json.Marshal(msg)
	if err != nil {
		return err
	}
	return wsutil.WriteClientText(c.Conn, data)
}

func generateToken() string {
	b := make([]byte, 50)
	rand.Read(b)
	return fmt.Sprintf("%x", b)
}

func logError(message string, err error) {
	fmt.Printf("%s: %v\n", message, err)
}

func SetupConfig(fileName string) Config {
	config := Config{
		Port:    defaultPort,
		SSL:     defaultSsl,
		Host:    hostName,
		BotName: botName,
	}
	if data, err := os.ReadFile(fileName); err == nil {
		json.Unmarshal(data, &config)
	}
	return config
}

func main() {
	config := SetupConfig(configFileName)
	client, err := NewClient(fmt.Sprintf("%s:%s", config.Host, config.Port), config.SSL)
	if err != nil {
		return
	}
	defer client.Close()

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("> ")
		scanner.Scan()
		text := strings.TrimSpace(scanner.Text())

		if text == "/quit" {
			return
		} else if strings.HasPrefix(text, "/lang") {
			parts := strings.Split(text, " ")
			if len(parts) == 2 {
				client.Locale = parts[1]
				fmt.Printf("Language changed to %s.\n", parts[1])
			} else {
				fmt.Println("Usage: /lang <locale>")
			}
		} else if text != "" {
			response, err := client.SendMessage(text)
			if err == nil {
				fmt.Printf("%s> %s\n", config.BotName, response.Content)
			}
		} else {
			fmt.Println("Please enter a message")
		}
	}
}
