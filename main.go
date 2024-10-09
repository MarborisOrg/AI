package main

import (
	"flag"
	"fmt"

	"github.com/gookit/color"
)

var neuralNetworksMain = map[string]Network{}

func main() {
	port := flag.String("port", "8080", "The port for the API and WebSocket.")
	flag.Parse()

	marborisASCII := string(ReadFile("res/marboris-ascii.txt"))
	fmt.Println(color.FgLightGreen.Render(marborisASCII))

	Authenticate()

	// en
	for _, locale := range Locales {
		SerializeMessages(locale.Tag)

		neuralNetworksMain[locale.Tag] = CreateNeuralNetwork(locale.Tag)
	}

	Serve(neuralNetworksMain, *port)
}
