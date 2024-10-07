package server

import (
	"fmt"
	"net/http"
	"time"

	"marboris/core/analysis"

	"marboris/core/dashboard"

	"marboris/core/network"

	"github.com/gookit/color"
	"github.com/gorilla/mux"
	gocache "github.com/patrickmn/go-cache"
)

var (
	// Create the neural network variable to use it everywhere
	neuralNetworks map[string]network.Network
	// Initializes the cache with a 5 minute lifetime
	cache = gocache.New(5*time.Minute, 5*time.Minute)
)

// Serve serves the server in the given port
func Serve(_neuralNetworks map[string]network.Network, port string) {
	// Set the current global network as a global variable
	neuralNetworks = _neuralNetworks // require

	// Initializes the router
	router := mux.NewRouter()
	// Serve the websocket
	router.HandleFunc("/websocket", SocketHandle)
	// Serve the API
	router.HandleFunc("/api/{locale}/dashboard", GetDashboardData).Methods("GET")
	router.HandleFunc("/api/{locale}/intent", dashboard.CreateIntent).Methods("POST")
	router.HandleFunc("/api/{locale}/intent", dashboard.DeleteIntent).Methods("DELETE", "OPTIONS")
	router.HandleFunc("/api/{locale}/intents", dashboard.GetIntents).Methods("GET")
	router.HandleFunc("/api/coverage", analysis.GetCoverage).Methods("GET")

	magenta := color.FgMagenta.Render
	fmt.Printf("\nServer listening on the port %s...\n", magenta(port))

	// Serves the chat
	err := http.ListenAndServe(":"+port, router)
	if err != nil {
		panic(err)
	}
}
