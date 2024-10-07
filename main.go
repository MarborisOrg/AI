package main

import (
	"flag"
	"fmt"

	"marboris/core/locales"
	"marboris/core/training"

	"marboris/core/dashboard"

	"marboris/core/util"

	"github.com/gookit/color"

	"marboris/core/network"

	"marboris/core/server"
)

var neuralNetworks = map[string]network.Network{}

func main() {
	port := flag.String("port", "8080", "The port for the API and WebSocket.")
	/*localesFlag := flag.String("re-train", "", "The locale(s) to re-train.")*/
	flag.Parse()

	// Print the Marboris ascii text
	marborisASCII := string(util.ReadFile("res/marboris-ascii.txt"))
	fmt.Println(color.FgLightGreen.Render(marborisASCII))

	// Create the authentication token
	dashboard.Authenticate()

	for _, locale := range locales.Locales {
		util.SerializeMessages(locale.Tag)

		neuralNetworks[locale.Tag] = training.CreateNeuralNetwork(
			locale.Tag,
			false,
		)
	}

	// Serves the server
	server.Serve(neuralNetworks, *port)
}
