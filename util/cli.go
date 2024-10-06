package util

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"time"
	"os"
)

// slice file

// Contains checks if a string slice contains a specified string
func Contains(slice []string, text string) bool {
	for _, item := range slice {
		if item == text {
			return true
		}
	}

	return false
}

// Difference returns the difference of slice and slice2
func Difference(slice []string, slice2 []string) (difference []string) {
	// Loop two times, first to find slice1 strings not in slice2,
	// second loop to find slice2 strings not in slice1
	for i := 0; i < 2; i++ {
		for _, s1 := range slice {
			found := false
			for _, s2 := range slice2 {
				if s1 == s2 {
					found = true
					break
				}
			}
			// String not found. We add it to return slice
			if !found {
				difference = append(difference, s1)
			}
		}
		// Swap the slices, only if it was the first loop
		if i == 0 {
			slice, slice2 = slice2, slice
		}
	}

	return difference
}

// Index returns a string index in a string slice
func Index(slice []string, text string) int {
	for i, item := range slice {
		if item == text {
			return i
		}
	}

	return 0
}


// messages file

// Message contains the message's tag and its contained matched sentences
type Message struct {
	Tag      string   `json:"tag"`
	Messages []string `json:"messages"`
}

var messages = map[string][]Message{}

// SerializeMessages serializes the content of `res/datasets/messages.json` in JSON
func SerializeMessages(locale string) []Message {
	var currentMessages []Message
	err := json.Unmarshal(ReadFile("res/locales/"+locale+"/messages.json"), &currentMessages)
	if err != nil {
		fmt.Println(err)
	}

	messages[locale] = currentMessages

	return currentMessages
}

// GetMessages returns the cached messages for the given locale
func GetMessages(locale string) []Message {
	return messages[locale]
}

// GetMessageByTag returns a message found by the given tag and locale
func GetMessageByTag(tag, locale string) Message {
	for _, message := range messages[locale] {
		if tag != message.Tag {
			continue
		}

		return message
	}

	return Message{}
}

// GetMessage retrieves a message tag and returns a random message chose from res/datasets/messages.json
func GetMessage(locale, tag string) string {
	for _, message := range messages[locale] {
		// Find the message with the right tag
		if message.Tag != tag {
			continue
		}

		// Returns the only element if there aren't more
		if len(message.Messages) == 1 {
			return message.Messages[0]
		}

		// Returns a random sentence
		rand.Seed(time.Now().UnixNano())
		return message.Messages[rand.Intn(len(message.Messages))]
	}

	return ""
}

// file

// ReadFile returns the bytes of a file searched in the path and beyond it
func ReadFile(path string) (bytes []byte) {
	bytes, err := os.ReadFile(path)
	if err != nil {
		bytes, err = os.ReadFile("../" + path)
	}

	if err != nil {
		panic(err)
	}

	return bytes
}
