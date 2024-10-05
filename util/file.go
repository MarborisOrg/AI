package util

import (
	"os"
)

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
