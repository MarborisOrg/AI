// =================================================================
package main

// =================================================================

// =================================================================
import (
	"bufio"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"

	"github.com/gorilla/websocket"
)

// =================================================================

// =================================================================
const (
	logFileName    = "logfile.log"
	configFileName = "config.json"
	defaultPort    = "8080"
	botName        = "marboris"
	hostName       = "localhost"
	defaultSsl     = false
)

var logChannel = make(chan string, 100) // Buffered channel for logs

// =================================================================

// =================================================================
type Client struct {
	Information *map[string]interface{}
	Locale      string
	Token       string
	Connection  *websocket.Conn
	Channel     chan string
	mu          sync.Mutex // Mutex for concurrent access
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
	Tag         string                 `json:"tag"`
	Information map[string]interface{} `json:"information"`
}

type Configuration struct {
	Port      string `json:"port"`
	Host      string `json:"host"`
	SSL       bool   `json:"ssl"`
	BotName   string `json:"bot_name"`
	UserToken string `json:"user_token"`
}

// =================================================================

// =================================================================
func NewClient(host string, ssl bool, information *map[string]interface{}) (*Client, error) {
	scheme := "ws"
	if ssl {
		scheme += "s"
	}
	url := fmt.Sprintf("%s://%s/websocket", scheme, host)
	connection, _, err := websocket.DefaultDialer.Dial(url, nil)
	if err != nil {
		writeLog(fmt.Sprintf("Failed to connect to websocket: %v", err))
		return nil, err
	}
	client := &Client{ // Create a pointer to Client
		Information: information,
		Locale:      "en",
		Token:       generateToken(),
		Connection:  connection,
		Channel:     make(chan string),
	}
	if err := client.handshake(); err != nil {
		writeLog(fmt.Sprintf("Handshake failed: %v", err))
		return nil, err
	}
	return client, nil // Return pointer to Client
}

func (client *Client) Close() {
	client.mu.Lock()
	defer client.mu.Unlock()
	writeLog("Closing websocket connection")
	client.Connection.Close()
}

func (client *Client) SendMessage(content string) (ResponseMessage, error) {
	client.mu.Lock()
	defer client.mu.Unlock()

	message := RequestMessage{
		Type:        1,
		Content:     content,
		Token:       client.Token,
		Information: *client.Information,
		Locale:      client.Locale,
	}

	bytes, err := json.Marshal(message)
	if err != nil {
		writeLog(fmt.Sprintf("Failed to marshal message: %v", err))
		return ResponseMessage{}, err
	}
	if err = client.Connection.WriteMessage(websocket.TextMessage, bytes); err != nil {
		writeLog(fmt.Sprintf("Failed to send message: %v", err))
		return ResponseMessage{}, err
	}

	_, bytes, err = client.Connection.ReadMessage()
	if err != nil {
		writeLog(fmt.Sprintf("Failed to read message: %v", err))
		return ResponseMessage{}, err
	}

	var response ResponseMessage
	if err := json.Unmarshal(bytes, &response); err != nil {
		writeLog(fmt.Sprintf("Failed to unmarshal response: %v", err))
		return ResponseMessage{}, err
	}
	return response, nil
}

func (client *Client) handshake() error {
	bytes, err := json.Marshal(RequestMessage{
		Type:        0,
		Content:     "",
		Token:       client.Token,
		Information: *client.Information,
	})
	if err != nil {
		writeLog(fmt.Sprintf("Failed to marshal handshake message: %v", err))
		return err
	}
	return client.Connection.WriteMessage(websocket.TextMessage, bytes)
}

func generateToken() string {
	b := make([]byte, 50)
	rand.Read(b)
	return fmt.Sprintf("%x", b)
}

func FileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func SetupConfig(fileName string) *Configuration {
	config := Configuration{
		Port:      defaultPort,
		SSL:       defaultSsl,
		Host:      hostName,
		BotName:   botName,
		UserToken: generateToken(),
	}

	if FileExists(fileName) {
		file, err := os.ReadFile(fileName)
		if err != nil {
			writeLog(fmt.Sprintf("Error reading config file: %v", err))
			return &config
		}
		if err := json.Unmarshal(file, &config); err != nil {
			writeLog(fmt.Sprintf("Error parsing config file: %v", err))
		}
	}

	fileData, err := json.MarshalIndent(config, "", " ")
	if err != nil {
		writeLog(fmt.Sprintf("Error marshaling config: %v", err))
		return &config
	}

	if err := os.WriteFile(fileName, fileData, 0644); err != nil {
		writeLog(fmt.Sprintf("Error writing config file: %v", err))
	}
	return &config
}

func writeLog(message string) {
	logChannel <- message // Send log message to channel
}

func logWriter() {
	var f *os.File
	var err error

	// Open the log file outside the loop
	f, err = os.OpenFile(logFileName, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		fmt.Printf("Could not open log file: %v\n", err)
		return
	}
	defer f.Close() // Ensure the file is closed when the function exits

	for msg := range logChannel {
		if _, err := f.WriteString(fmt.Sprintf("%s\n", msg)); err != nil {
			fmt.Printf("Could not write to log file: %v\n", err)
		}
	}
}

func main() {
	// Start log writer goroutine
	go logWriter()

	// Handle graceful shutdown
	setupGracefulShutdown()

	config := SetupConfig(configFileName)

	var information map[string]interface{}
	client, err := NewClient(fmt.Sprintf("%s:%s", config.Host, config.Port), config.SSL, &information)
	if err != nil {
		writeLog(fmt.Sprintf("Error creating client: %v", err))
		return
	}
	defer client.Close()

	fmt.Println("Enter message to " + config.BotName + " or type:")
	fmt.Printf("- /quit to quit\n")
	fmt.Printf("- /lang <locale> to change the language\n\n")

	scanner := bufio.NewScanner(os.Stdin)

	for {
		fmt.Print("> ")
		scanner.Scan()
		text := scanner.Text()

		switch {
		case strings.TrimSpace(text) == "":
			writeLog("Empty message entered")
			fmt.Println("Please enter a message")
		case text == "/quit" || text == "/q" || text == ":q":
			writeLog("Quitting the application")
			return
		case strings.HasPrefix(text, "/lang"):
			arguments := strings.Split(text, " ")[1:]
			if len(arguments) != 1 {
				writeLog("Wrong number of arguments for language command")
				fmt.Println("Wrong number of arguments, language command should contain only the locale")
			} else {
				client.Locale = arguments[0]
				fmt.Printf("Language changed to %s.\n", arguments[0])
			}
		default:
			response, err := client.SendMessage(text)
			if err == nil {
				writeLog(fmt.Sprintf("Message sent: %s, Response: %s", text, response.Content))
				fmt.Printf("%s> %s\n", config.BotName, response.Content)
			} else {
				writeLog(fmt.Sprintf("Error sending message: %v", err))
			}
		}
	}
}

func setupGracefulShutdown() {
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigs
		writeLog("Shutting down server gracefully...")
		close(logChannel)
		os.Exit(0)
	}()
}

// =================================================================
