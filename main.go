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
	flag.Parse()

	marborisASCII := string(cli.ReadFile("res/marboris-ascii.txt"))
	fmt.Println(color.FgLightGreen.Render(marborisASCII))

	cli.Authenticate()

	// en
	for _, locale := range cli.Locales {
		cli.SerializeMessages(locale.Tag)

		neuralNetworks[locale.Tag] = cli.CreateNeuralNetwork(locale.Tag)
	}

	cli.Serve(neuralNetworks, *port)
}
