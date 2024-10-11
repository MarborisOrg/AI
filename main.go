package main

import (
	"flag"
	"fmt"

	"github.com/gookit/color"
	"marboris/src"
)

var neuralNetworksMain = map[string]core.Network{}

func main() {
	port := flag.String("port", "8080", "The port for the API and WebSocket.")
	flag.Parse()

	marborisASCII := string(core.ReadFile("res/marboris-ascii.txt"))
	fmt.Println(color.FgLightGreen.Render(marborisASCII))

	core.Authenticate()

	// en
	for _, locale := range core.Locales {
		core.SerializeMessages(locale.Tag)

		neuralNetworksMain[locale.Tag] = core.CreateNeuralNetwork(locale.Tag)
	}

	core.Serve(neuralNetworksMain, *port)
}
