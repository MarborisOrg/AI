package main

import (
	"flag"
	"fmt"

	"github.com/gookit/color"

	"marboris/core/cli"
	"marboris/core/trainc"
)

var neuralNetworks = map[string]cli.Network{}

func main() {
	port := flag.String("port", "8080", "The port for the API and WebSocket.")
	flag.Parse()

	// Print the Marboris ascii text
	marborisASCII := string(cli.ReadFile("res/marboris-ascii.txt"))
	fmt.Println(color.FgLightGreen.Render(marborisASCII))

	// Create the authentication token
	cli.Authenticate()

	for _, locale := range cli.Locales {
		cli.SerializeMessages(locale.Tag)

		trainc.CreateNeuralNetwork(
			locale.Tag,
			false,
		)
		neuralNetworks[locale.Tag] = cli.CreateNeuralNetwork(
			locale.Tag,
			false,
		)
	}

	// Serves the server
	cli.Serve(neuralNetworks, *port)
}
