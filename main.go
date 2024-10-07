package main

import (
	"flag"
	"fmt"

	"github.com/gookit/color"

	"marboris/core/cli"
)

var neuralNetworks = map[string]cli.Network{}

func main() {
	port := flag.String("port", "8080", "The port for the API and WebSocket.")
	/*localesFlag := flag.String("re-train", "", "The locale(s) to re-train.")*/
	flag.Parse()

	// Print the Marboris ascii text
	marborisASCII := string(cli.ReadFile("res/marboris-ascii.txt"))
	fmt.Println(color.FgLightGreen.Render(marborisASCII))

	// Create the authentication token
	cli.Authenticate()

	for _, locale := range cli.Locales {
		cli.SerializeMessages(locale.Tag)

		neuralNetworks[locale.Tag] = cli.CreateNeuralNetwork(
			locale.Tag,
			false,
		)
	}

	// Serves the server
	cli.Serve(neuralNetworks, *port)
}
