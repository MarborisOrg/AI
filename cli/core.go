package cli

import (
	"encoding/json"
	"github.com/gookit/color"

	"errors"
	"fmt"
	"math/rand"
	"os"
	"sort"
	"time"

	"net/http"
	"regexp"
	"strings"

	"github.com/soudy/mathcat"
	"reflect"

	"bufio"
	"encoding/csv"
	"github.com/gorilla/websocket"
	"golang.org/x/crypto/bcrypt"
	"gopkg.in/cheggaaa/pb.v1"
	"io"
	"math"
	"strconv"

	"log"

	"github.com/gorilla/mux"
	"github.com/tebeka/snowball"

	gocache "github.com/patrickmn/go-cache"
)

// package util

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
func GetMessageu(locale, tag string) string {
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

// package user

// Information is the user's information retrieved from the client
type Information struct {
	Name           string   `json:"name"`
	MovieGenres    []string `json:"movie_genres"`
	MovieBlacklist []string `json:"movie_blacklist"`
}

// userInformation is a map which is the cache for user information
var userInformation = map[string]Information{}

// ChangeUserInformation requires the token of the user and a function which gives the actual
// information and returns the new information.
func ChangeUserInformation(token string, changer func(Information) Information) {
	userInformation[token] = changer(userInformation[token])
}

// SetUserInformation sets the user's information by its token.
func SetUserInformation(token string, information Information) {
	userInformation[token] = information
}

// GetUserInformation returns the information of a user with his token
func GetUserInformation(token string) Information {
	return userInformation[token]
}

// package training

// TrainData returns the inputs and outputs for the neural network
func TrainData(locale string) (inputs, outputs [][]float64) {
	words, classes, documents := Organize(locale)

	for _, document := range documents {
		outputRow := make([]float64, len(classes))
		bag := document.Sentence.WordsBag(words)

		// Change value to 1 where there is the document Tag
		outputRow[Index(classes, document.Tag)] = 1

		// Append data to inputs and outputs
		inputs = append(inputs, bag)
		outputs = append(outputs, outputRow)
	}

	return inputs, outputs
}

// CreateNeuralNetwork returns a new neural network which is loaded from res/training.json or
// trained from TrainData() inputs and targets.
func CreateNeuralNetwork(locale string, ignoreTrainingFile bool) (neuralNetwork Network) {
	// Decide if the network is created by the save or is a new one
	saveFile := "res/locales/" + locale + "/training.json"

	// _, err := os.Open(saveFile)
	// Train the model if there is no training file
	if true { // err != nil || ignoreTrainingFile
		inputs, outputs := TrainData(locale)

		neuralNetwork = CreateNetwork(locale, 0.1, inputs, outputs, 50)
		neuralNetwork.Train(200)

		// Save the neural network in res/training.json
		neuralNetwork.Save(saveFile)
	} else {
		fmt.Printf(
			"%s %s\n",
			color.FgBlue.Render("Loading the neural network from"),
			color.FgRed.Render(saveFile),
		)
		// Initialize the intents
		SerializeIntents(locale)
		neuralNetwork = *LoadNetwork(saveFile)
	}

	return
}

// package network

/**
	derivative
**/

// Derivative contains the derivatives of `z` and the adjustments
type Derivative struct {
	Delta      Matrix
	Adjustment Matrix
}

// ComputeLastLayerDerivatives returns the derivatives of the last layer L
func (network Network) ComputeLastLayerDerivatives() Derivative {
	l := len(network.Layers) - 1
	lastLayer := network.Layers[l]

	// Compute derivative for the last layer of weights and biases
	cost := Differencen(network.Output, lastLayer)
	sigmoidDerivative := Multiplication(lastLayer, ApplyFunction(lastLayer, SubtractsOne))

	// Compute delta and the weights' adjustment
	delta := Multiplication(
		ApplyFunction(cost, MultipliesByTwo),
		sigmoidDerivative,
	)
	weights := DotProduct(Transpose(network.Layers[l-1]), delta)

	return Derivative{
		Delta:      delta,
		Adjustment: weights,
	}
}

// ComputeDerivatives returns the derivatives of a specific layer l defined by i
func (network Network) ComputeDerivatives(i int, derivatives []Derivative) Derivative {
	l := len(network.Layers) - 2 - i

	// Compute derivative for the layer of weights and biases
	delta := Multiplication(
		DotProduct(
			derivatives[i].Delta,
			Transpose(network.Weights[l]),
		),
		Multiplication(
			network.Layers[l],
			ApplyFunction(network.Layers[l], SubtractsOne),
		),
	)
	weights := DotProduct(Transpose(network.Layers[l-1]), delta)

	return Derivative{
		Delta:      delta,
		Adjustment: weights,
	}
}

// Adjust make the adjusts
func (network Network) Adjust(derivatives []Derivative) {
	for i, derivative := range derivatives {
		l := len(derivatives) - i

		network.Weights[l-1] = Sum(
			network.Weights[l-1],
			ApplyRate(derivative.Adjustment, network.Rate),
		)
		network.Biases[l-1] = Sum(
			network.Biases[l-1],
			ApplyRate(derivative.Delta, network.Rate),
		)
	}
}

/**
	math
**/

// Sigmoid is the activation function
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

// MultipliesByTwo takes a float and returns the float multiplied by two
func MultipliesByTwo(x float64) float64 {
	return 2 * x
}

// SubtractsOne takes a float and returns the float subtracted by one
func SubtractsOne(x float64) float64 {
	return x - 1
}

/**
	matrix
**/

// Matrix is an alias for [][]float64
type Matrix [][]float64

// RandomMatrix returns the value of a random matrix of *rows* and *columns* dimensions and
// where the values are between *lower* and *upper*.
func RandomMatrix(rows, columns int) (matrix Matrix) {
	matrix = make(Matrix, rows)

	for i := 0; i < rows; i++ {
		matrix[i] = make([]float64, columns)
		for j := 0; j < columns; j++ {
			matrix[i][j] = rand.Float64()*2.0 - 1.0
		}
	}

	return
}

// CreateMatrix returns an empty matrix which is the size of rows and columns
func CreateMatrix(rows, columns int) (matrix Matrix) {
	matrix = make(Matrix, rows)

	for i := 0; i < rows; i++ {
		matrix[i] = make([]float64, columns)
	}

	return
}

// Rows returns number of matrix's rows
func Rows(matrix Matrix) int {
	return len(matrix)
}

// Columns returns number of matrix's columns
func Columns(matrix Matrix) int {
	return len(matrix[0])
}

// ApplyFunctionWithIndex returns a matrix where fn has been applied with the indexes provided
func ApplyFunctionWithIndex(matrix Matrix, fn func(i, j int, x float64) float64) Matrix {
	for i := 0; i < Rows(matrix); i++ {
		for j := 0; j < Columns(matrix); j++ {
			matrix[i][j] = fn(i, j, matrix[i][j])
		}
	}

	return matrix
}

// ApplyFunction returns a matrix where fn has been applied
func ApplyFunction(matrix Matrix, fn func(x float64) float64) Matrix {
	return ApplyFunctionWithIndex(matrix, func(i, j int, x float64) float64 {
		return fn(x)
	})
}

// ApplyRate returns a matrix where the learning rate has been multiplies
func ApplyRate(matrix Matrix, rate float64) Matrix {
	return ApplyFunction(matrix, func(x float64) float64 {
		return rate * x
	})
}

// DotProduct returns a matrix which is the result of the dot product between matrix and matrix2
func DotProduct(matrix, matrix2 Matrix) Matrix {
	if Columns(matrix) != Rows(matrix2) {
		panic("Cannot make dot product between these two matrix.")
	}

	return ApplyFunctionWithIndex(
		CreateMatrix(Rows(matrix), Columns(matrix2)),
		func(i, j int, x float64) float64 {
			var sum float64

			for k := 0; k < Columns(matrix); k++ {
				sum += matrix[i][k] * matrix2[k][j]
			}

			return sum
		},
	)
}

// Sum returns the sum of matrix and matrix2
func Sum(matrix, matrix2 Matrix) (resultMatrix Matrix) {
	ErrorNotSameSize(matrix, matrix2)

	// do not change
	resultMatrix = CreateMatrix(Rows(matrix), Columns(matrix))

	return ApplyFunctionWithIndex(matrix, func(i, j int, x float64) float64 {
		return matrix[i][j] + matrix2[i][j]
	})
}

// Difference returns the difference between matrix and matrix2
func Differencen(matrix, matrix2 Matrix) (resultMatrix Matrix) {
	ErrorNotSameSize(matrix, matrix2)

	// do not change
	resultMatrix = CreateMatrix(Rows(matrix), Columns(matrix))

	return ApplyFunctionWithIndex(resultMatrix, func(i, j int, x float64) float64 {
		return matrix[i][j] - matrix2[i][j]
	})
}

// Multiplication returns the multiplication of matrix and matrix2
func Multiplication(matrix, matrix2 Matrix) (resultMatrix Matrix) {
	ErrorNotSameSize(matrix, matrix2)

	// do not change
	resultMatrix = CreateMatrix(Rows(matrix), Columns(matrix))

	return ApplyFunctionWithIndex(matrix, func(i, j int, x float64) float64 {
		return matrix[i][j] * matrix2[i][j]
	})
}

// Transpose returns the matrix transposed
func Transpose(matrix Matrix) (resultMatrix Matrix) {
	// do not change
	resultMatrix = CreateMatrix(Columns(matrix), Rows(matrix))

	for i := 0; i < Rows(matrix); i++ {
		for j := 0; j < Columns(matrix); j++ {
			resultMatrix[j][i] = matrix[i][j]
		}
	}

	return resultMatrix
}

// ErrorNotSameSize panics if the matrices do not have the same dimension
func ErrorNotSameSize(matrix, matrix2 Matrix) {
	if Rows(matrix) != Rows(matrix2) && Columns(matrix) != Columns(matrix2) {
		panic("These two matrices must have the same dimension.")
	}
}

/**
	network
**/

// Network contains the Layers, Weights, Biases of a neural network then the actual output values
// and the learning rate.
type Network struct {
	Layers  []Matrix
	Weights []Matrix
	Biases  []Matrix
	Output  Matrix
	Rate    float64
	Errors  []float64
	Time    float64
	Locale  string
}

// LoadNetwork returns a Network from a specified file
func LoadNetwork(fileName string) *Network {
	inF, err := os.Open(fileName)
	if err != nil {
		panic("Failed to load " + fileName + ".")
	}
	defer inF.Close()

	decoder := json.NewDecoder(inF)
	neuralNetwork := &Network{}
	err = decoder.Decode(neuralNetwork)
	if err != nil {
		panic(err)
	}

	return neuralNetwork
}

// CreateNetwork creates the network by generating the layers, weights and biases
func CreateNetwork(locale string, rate float64, input, output Matrix, hiddensNodes ...int) Network {
	input = append([][]float64{
		make([]float64, len(input[0])),
	}, input...)
	output = append([][]float64{
		make([]float64, len(output[0])),
	}, output...)

	// Create the layers arrays and add the input values
	inputMatrix := input
	layers := []Matrix{inputMatrix}
	// Generate the hidden layer
	for _, hiddenNodes := range hiddensNodes {
		layers = append(layers, CreateMatrix(len(input), hiddenNodes))
	}
	// Add the output values to the layers arrays
	layers = append(layers, output)

	// Generate the weights and biases
	weightsNumber := len(layers) - 1
	var weights []Matrix
	var biases []Matrix

	for i := 0; i < weightsNumber; i++ {
		rows, columns := Columns(layers[i]), Columns(layers[i+1])

		weights = append(weights, RandomMatrix(rows, columns))
		biases = append(biases, RandomMatrix(Rows(layers[i]), columns))
	}

	return Network{
		Layers:  layers,
		Weights: weights,
		Biases:  biases,
		Output:  output,
		Rate:    rate,
		Locale:  locale,
	}
}

// Save saves the neural network in a specified file which can be retrieved with LoadNetwork
func (network Network) Save(fileName string) {
	outF, err := os.OpenFile(fileName, os.O_CREATE|os.O_RDWR, 0777)
	if err != nil {
		panic("Failed to save the network to " + fileName + ".")
	}
	defer outF.Close()

	encoder := json.NewEncoder(outF)
	err = encoder.Encode(network)
	if err != nil {
		panic(err)
	}
}

// FeedForward executes forward propagation for the given inputs in the network
func (network *Network) FeedForward() {
	for i := 0; i < len(network.Layers)-1; i++ {
		layer, weights, biases := network.Layers[i], network.Weights[i], network.Biases[i]

		productMatrix := DotProduct(layer, weights)
		Sum(productMatrix, biases)
		ApplyFunction(productMatrix, Sigmoid)

		// Replace the output values
		network.Layers[i+1] = productMatrix
	}
}

// Predict returns the predicted value for a training example
func (network *Network) Predict(input []float64) []float64 {
	network.Layers[0] = Matrix{input}
	network.FeedForward()
	return network.Layers[len(network.Layers)-1][0]
}

// FeedBackward executes back propagation to adjust the weights for all the layers
func (network *Network) FeedBackward() {
	var derivatives []Derivative
	derivatives = append(derivatives, network.ComputeLastLayerDerivatives())

	// Compute the derivatives of the hidden layers
	for i := 0; i < len(network.Layers)-2; i++ {
		derivatives = append(derivatives, network.ComputeDerivatives(i, derivatives))
	}

	// Then adjust the weights and biases
	network.Adjust(derivatives)
}

// ComputeError returns the average of all the errors after the training
func (network *Network) ComputeError() float64 {
	// Feed forward to compute the last layer's values
	network.FeedForward()
	lastLayer := network.Layers[len(network.Layers)-1]
	errors := Differencen(network.Output, lastLayer)

	// Make the sum of all the errors
	var i int
	var sum float64
	for _, a := range errors {
		for _, e := range a {
			sum += e
			i++
		}
	}

	// Compute the average
	return sum / float64(i)
}

// Train trains the neural network with a given number of iterations by executing
// forward and back propagation
func (network *Network) Train(iterations int) {
	// Initialize the start date
	start := time.Now()

	// Create the progress bar
	bar := pb.New(iterations).Postfix(fmt.Sprintf(
		" - %s %s %s",
		color.FgBlue.Render("Training the"),
		color.FgRed.Render("english"), // locales.GetNameByTag(network.Locale)
		color.FgBlue.Render("neural network"),
	))
	bar.Format("(██░)")
	bar.SetMaxWidth(60)
	bar.ShowCounters = false
	bar.Start()

	// Train the network
	for i := 0; i < iterations; i++ {
		network.FeedForward()
		network.FeedBackward()

		// Append errors for dashboard data
		if i%(iterations/20) == 0 {
			network.Errors = append(
				network.Errors,
				// Round the error to two decimals
				network.ComputeError(),
			)
		}

		// Increment the progress bar
		bar.Increment()
	}

	bar.Finish()
	// Print the error
	arrangedError := fmt.Sprintf("%.5f", network.ComputeError())

	// Calculate elapsed date
	elapsed := time.Since(start)
	// Round the elapsed date at two decimals
	network.Time = math.Floor(elapsed.Seconds()*100) / 100

	fmt.Printf("The error rate is %s.\n", color.FgGreen.Render(arrangedError))
}

// package analysis

/**
	sentence
**/
// A Sentence represents simply a sentence with its content as a string
type Sentence struct {
	Locale  string
	Content string
}

// Result contains a predicted value with its tag and its value
type Result struct {
	Tag   string
	Value float64
}

var userCache = gocache.New(5*time.Minute, 5*time.Minute)

// DontUnderstand contains the tag for the don't understand messages
const DontUnderstand = "don't understand"

// NewSentence returns a Sentence object where the content has been arranged
func NewSentence(locale, content string) (sentence Sentence) {
	sentence = Sentence{
		Locale:  locale,
		Content: content,
	}
	sentence.arrange()

	return
}

// PredictTag classifies the sentence with the model
func (sentence Sentence) PredictTag(neuralNetwork Network) string {
	words, classes, _ := Organize(sentence.Locale)

	// Predict with the model
	predict := neuralNetwork.Predict(sentence.WordsBag(words))

	// Enumerate the results with the intent tags
	var resultsTag []Result
	for i, result := range predict {
		if i >= len(classes) {
			continue
		}
		resultsTag = append(resultsTag, Result{classes[i], result})
	}

	// Sort the results in ascending order
	sort.Slice(resultsTag, func(i, j int) bool {
		return resultsTag[i].Value > resultsTag[j].Value
	})

	LogResults(sentence.Locale, sentence.Content, resultsTag)

	return resultsTag[0].Tag
}

// RandomizeResponse takes the entry message, the response tag and the token and returns a random
// message from res/datasets/intents.json where the triggers are applied
func RandomizeResponse(locale, entry, tag, token string) (string, string) {
	if tag == DontUnderstand {
		return DontUnderstand, GetMessageu(locale, tag)
	}

	// Append the modules intents to the intents from res/datasets/intents.json
	intents := append(SerializeIntents(locale), SerializeModulesIntents(locale)...)

	for _, intent := range intents {
		if intent.Tag != tag {
			continue
		}

		// Reply a "don't understand" message if the context isn't correct
		cacheTag, _ := userCache.Get(token)
		if intent.Context != "" && cacheTag != intent.Context {
			return DontUnderstand, GetMessageu(locale, DontUnderstand)
		}

		// Set the actual context
		userCache.Set(token, tag, gocache.DefaultExpiration)

		// Choose a random response in intents
		response := intent.Responses[0]
		if len(intent.Responses) > 1 {
			rand.Seed(time.Now().UnixNano())
			response = intent.Responses[rand.Intn(len(intent.Responses))]
		}

		// And then apply the triggers on the message
		return ReplaceContent(locale, tag, entry, response, token)
	}

	return DontUnderstand, GetMessageu(locale, DontUnderstand)
}

// Calculate send the sentence content to the neural network and returns a response with the matching tag
func (sentence Sentence) Calculate(cache gocache.Cache, neuralNetwork Network, token string) (string, string) {
	tag, found := cache.Get(sentence.Content)

	// Predict tag with the neural network if the sentence isn't in the cache
	if !found {
		tag = sentence.PredictTag(neuralNetwork)
		cache.Set(sentence.Content, tag, gocache.DefaultExpiration)
	}

	return RandomizeResponse(sentence.Locale, sentence.Content, tag.(string), token)
}

// LogResults print in the console the sentence and its tags sorted by prediction
func LogResults(locale, entry string, results []Result) {
	// If NO_LOGS is present, then don't print the given messages
	if os.Getenv("NO_LOGS") == "1" {
		return
	}

	green := color.FgGreen.Render
	yellow := color.FgYellow.Render

	fmt.Printf(
		"\n“%s” - %s\n",
		color.FgCyan.Render(entry),
		color.FgRed.Render(GetNameByTag(locale)),
	)
	for _, result := range results {
		// Arbitrary choice of 0.004 to have less tags to show
		if result.Value < 0.004 {
			continue
		}

		fmt.Printf("  %s %s - %s\n", green("▫︎"), result.Tag, yellow(result.Value))
	}
}

/**
	intents
**/

var intents = map[string][]Intent{}

// Intent is a way to group sentences that mean the same thing and link them with a tag which
// represents what they mean, some responses that the bot can reply and a context
type Intent struct {
	Tag       string   `json:"tag"`
	Patterns  []string `json:"patterns"`
	Responses []string `json:"responses"`
	Context   string   `json:"context"`
}

// Document is any sentence from the intents' patterns linked with its tag
type Document struct {
	Sentence Sentence
	Tag      string
}

// CacheIntents set the given intents to the global variable intents
func CacheIntents(locale string, _intents []Intent) {
	intents[locale] = _intents
}

// GetIntents returns the cached intents
func GetIntentsa(locale string) []Intent {
	return intents[locale]
}

// SerializeIntents returns a list of intents retrieved from the given intents file
func SerializeIntents(locale string) (_intents []Intent) {
	err := json.Unmarshal(ReadFile("res/locales/"+locale+"/intents.json"), &_intents)
	if err != nil {
		panic(err)
	}

	CacheIntents(locale, _intents)

	return _intents
}

// SerializeModulesIntents retrieves all the registered modules and returns an array of Intents
func SerializeModulesIntents(locale string) []Intent {
	registeredModules := GetModules(locale)
	intents := make([]Intent, len(registeredModules))

	for k, module := range registeredModules {
		intents[k] = Intent{
			Tag:       module.Tag,
			Patterns:  module.Patterns,
			Responses: module.Responses,
			Context:   "",
		}
	}

	return intents
}

// GetIntentByTag returns an intent found by given tag and locale
func GetIntentByTag(tag, locale string) Intent {
	for _, intent := range GetIntentsa(locale) {
		if tag != intent.Tag {
			continue
		}

		return intent
	}

	return Intent{}
}

// Organize intents with an array of all words, an array with a representative word of each tag
// and an array of Documents which contains a word list associated with a tag
func Organize(locale string) (words, classes []string, documents []Document) {
	// Append the modules intents to the intents from res/datasets/intents.json
	intents := append(
		SerializeIntents(locale),
		SerializeModulesIntents(locale)...,
	)

	for _, intent := range intents {
		for _, pattern := range intent.Patterns {
			// Tokenize the pattern's sentence
			patternSentence := Sentence{locale, pattern}
			patternSentence.arrange()

			// Add each word to response
			for _, word := range patternSentence.stem() {

				if !Contains(words, word) {
					words = append(words, word)
				}
			}

			// Add a new document
			documents = append(documents, Document{
				patternSentence,
				intent.Tag,
			})
		}

		// Add the intent tag to classes
		classes = append(classes, intent.Tag)
	}

	sort.Strings(words)
	sort.Strings(classes)

	return words, classes, documents
}

/**
	format
**/

// arrange checks the format of a string to normalize it, remove ignored characters
func (sentence *Sentence) arrange() {
	// Remove punctuation after letters
	punctuationRegex := regexp.MustCompile(`[a-zA-Z]( )?(\.|\?|!|¿|¡)`)
	sentence.Content = punctuationRegex.ReplaceAllStringFunc(sentence.Content, func(s string) string {
		punctuation := regexp.MustCompile(`(\.|\?|!)`)
		return punctuation.ReplaceAllString(s, "")
	})

	sentence.Content = strings.ReplaceAll(sentence.Content, "-", " ")
	sentence.Content = strings.TrimSpace(sentence.Content)
}

// removeStopWords takes an arary of words, removes the stopwords and returns it
func removeStopWords(locale string, words []string) []string {
	// Don't remove stopwords for small sentences like “How are you” because it will remove all the words
	if len(words) <= 4 {
		return words
	}

	// Read the content of the stopwords file
	stopWords := string(ReadFile("res/locales/" + locale + "/stopwords.txt"))

	var wordsToRemove []string

	// Iterate through all the stopwords
	for _, stopWord := range strings.Split(stopWords, "\n") {
		// Iterate through all the words of the given array
		for _, word := range words {
			// Continue if the word isn't a stopword
			if !strings.Contains(stopWord, word) {
				continue
			}

			wordsToRemove = append(wordsToRemove, word)
		}
	}

	return Difference(words, wordsToRemove)
}

// tokenize returns a list of words that have been lower-cased
func (sentence Sentence) tokenize() (tokens []string) {
	// Split the sentence in words
	tokens = strings.Fields(sentence.Content)

	// Lower case each word
	for i, token := range tokens {
		tokens[i] = strings.ToLower(token)
	}

	tokens = removeStopWords(sentence.Locale, tokens)

	return
}

// stem returns the sentence split in stemmed words
func (sentence Sentence) stem() (tokenizeWords []string) {
	locale := GetTagByName(sentence.Locale)
	// Set default locale to english
	if locale == "" {
		locale = "english"
	}

	tokens := sentence.tokenize()

	stemmer, err := snowball.New(locale)
	if err != nil {
		fmt.Println("Stemmer error", err)
		return
	}

	// Get the string token and push it to tokenizeWord
	for _, tokenizeWord := range tokens {
		word := stemmer.Stem(tokenizeWord)
		tokenizeWords = append(tokenizeWords, word)
	}

	return
}

// WordsBag retrieves the intents words and returns the sentence converted in a bag of words
func (sentence Sentence) WordsBag(words []string) (bag []float64) {
	for _, word := range words {
		// Append 1 if the patternWords contains the actual word, else 0
		var valueToAppend float64
		if Contains(sentence.stem(), word) {
			valueToAppend = 1
		}

		bag = append(bag, valueToAppend)
	}

	return bag
}

/**
	coverage
**/

var (
	defaultModules  []Modulem
	defaultIntents  []Intent
	defaultMessages []Message
)

// LocaleCoverage is the element for the coverage of each language
type LocaleCoverage struct {
	Tag      string   `json:"locale_tag"`
	Language string   `json:"language"`
	Coverage Coverage `json:"coverage"`
}

// Coverage is the coverage for a single language which contains the coverage details of each section
type Coverage struct {
	Modules  CoverageDetails `json:"modules"`
	Intents  CoverageDetails `json:"intents"`
	Messages CoverageDetails `json:"messages"`
}

// CoverageDetails are the details of items not covered and the coverage percentage
type CoverageDetails struct {
	NotCovered []string `json:"not_covered"`
	Coverage   int      `json:"coverage"`
}

// GetCoverage encodes the coverage of each language in json
func GetCoverage(writer http.ResponseWriter, _ *http.Request) {
	allowedHeaders := "Accept, Content-Type, Content-Length, Accept-Encoding, Authorization,Marboris-Token"
	writer.Header().Set("Access-Control-Allow-Origin", "*")
	writer.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
	writer.Header().Set("Access-Control-Allow-Headers", allowedHeaders)
	writer.Header().Set("Access-Control-Expose-Headers", "Authorization")

	defaultMessages, defaultIntents, defaultModules =
		GetMessages("en"), GetIntentsa("en"), GetModules("en")

	var coverage []LocaleCoverage

	// Calculate coverage for each language
	for _, locale := range Locales {
		if locale.Tag == "en" {
			continue
		}

		coverage = append(coverage, LocaleCoverage{
			Tag:      locale.Tag,
			Language: GetNameByTag(locale.Tag),
			Coverage: Coverage{
				Modules:  getModuleCoverage(locale.Tag),
				Intents:  getIntentCoverage(locale.Tag),
				Messages: getMessageCoverage(locale.Tag),
			},
		})
	}

	json.NewEncoder(writer).Encode(coverage)
}

// getMessageCoverage returns an array of not covered messages and the percentage of message that
// aren't translated in the given locale.
func getMessageCoverage(locale string) CoverageDetails {
	var notCoveredMessages []string

	// Iterate through the default messages which are the english ones to verify if a message isn't
	// translated in the given locale.
	for _, defaultMessage := range defaultMessages {
		message := GetMessageByTag(defaultMessage.Tag, locale)

		// Add the current module tag to the list of not-covered-modules
		if message.Tag != defaultMessage.Tag {
			notCoveredMessages = append(notCoveredMessages, defaultMessage.Tag)
		}
	}

	// Calculate the percentage of modules that aren't translated in the given locale
	coverage := calculateCoverage(len(notCoveredMessages), len(defaultMessages))

	return CoverageDetails{
		NotCovered: notCoveredMessages,
		Coverage:   coverage,
	}
}

// getIntentCoverage returns an array of not covered intents and the percentage of intents that aren't
// translated in the given locale.
func getIntentCoverage(locale string) CoverageDetails {
	var notCoveredIntents []string

	// Iterate through the default intents which are the english ones to verify if an intent isn't
	// translated in the given locale.
	for _, defaultIntent := range defaultIntents {
		intent := GetIntentByTag(defaultIntent.Tag, locale)

		// Add the current module tag to the list of not-covered-modules
		if intent.Tag != defaultIntent.Tag {
			notCoveredIntents = append(notCoveredIntents, defaultIntent.Tag)
		}
	}

	// Calculate the percentage of modules that aren't translated in the given locale
	coverage := calculateCoverage(len(notCoveredIntents), len(defaultModules))

	return CoverageDetails{
		NotCovered: notCoveredIntents,
		Coverage:   coverage,
	}
}

// getModuleCoverage returns an array of not covered modules and the percentage of modules that aren't
// translated in the given locale.
func getModuleCoverage(locale string) CoverageDetails {
	var notCoveredModules []string

	// Iterate through the default modules which are the english ones to verify if a module isn't
	// translated in the given locale.
	for _, defaultModule := range defaultModules {
		module := GetModuleByTag(defaultModule.Tag, locale)

		// Add the current module tag to the list of not-covered-modules
		if module.Tag != defaultModule.Tag {
			notCoveredModules = append(notCoveredModules, defaultModule.Tag)
		}
	}

	// Calculate the percentage of modules that aren't translated in the given locale
	coverage := calculateCoverage(len(notCoveredModules), len(defaultModules))

	return CoverageDetails{
		NotCovered: notCoveredModules,
		Coverage:   coverage,
	}
}

// calculateCoverage returns the coverage calculated with the given length of not covered
// items and the default items length
func calculateCoverage(notCoveredLength, defaultLength int) int {
	return 100 * (defaultLength - notCoveredLength) / defaultLength
}

// package locales

// Locales is the list of locales's tags and names
// Please check if the language is supported in https://github.com/tebeka/snowball,
// if it is please add the correct language name.
var Locales = []Locale{
	{
		Tag:  "en",
		Name: "english",
	},
}

// A Locale is a registered locale in the file
type Locale struct {
	Tag  string
	Name string
}

// GetNameByTag returns the name of the given locale's tag
func GetNameByTag(tag string) string {
	for _, locale := range Locales {
		if locale.Tag != tag {
			continue
		}

		return locale.Name
	}

	return ""
}

// GetTagByName returns the tag of the given locale's name
func GetTagByName(name string) string {
	for _, locale := range Locales {
		if locale.Name != name {
			continue
		}

		return locale.Tag
	}

	return ""
}

// Exists checks if the given tag exists in the list of locales
func Exists(tag string) bool {
	for _, locale := range Locales {
		if locale.Tag == tag {
			return true
		}
	}

	return false
}

// package dashboard

var fileName = "res/authentication.txt"

var authenticationHash []byte

// GenerateToken generates a random token of 30 characters and returns it
func GenerateToken() string {
	b := make([]byte, 30)
	rand.Read(b)

	fmt.Println("hey")
	return fmt.Sprintf("%x", b)
}

// HashToken gets the given tokens and returns its hash using bcrypt
func HashToken(token string) []byte {
	bytes, _ := bcrypt.GenerateFromPassword([]byte(token), 14)
	return bytes
}

// ChecksToken checks if the given token is the good one from the authentication file
func ChecksToken(token string) bool {
	err := bcrypt.CompareHashAndPassword(authenticationHash, []byte(token))
	return err == nil
}

// AuthenticationFileExists checks if the authentication file exists and return the condition
func AuthenticationFileExists() bool {
	_, err := os.Open(fileName)
	return err == nil
}

// SaveHash saves the given hash to the authentication file
func SaveHash(hash string) {
	file, err := os.Create(fileName)
	if err != nil {
		panic(err)
	}

	defer file.Close()

	file.WriteString(hash)
}

// Authenticate checks if the authentication file exists and if not it generates the file with a new token
func Authenticate() {
	// Do nothing if the authentication file exists
	if AuthenticationFileExists() {
		authenticationHash = ReadFile(fileName)
		return
	}

	// Generates the token and gives it to the user
	token := GenerateToken()
	fmt.Printf("Your authentication token is: %s\n", color.FgLightGreen.Render(token))
	fmt.Println("Save it, you won't be able to get it again unless you generate a new one.")
	fmt.Println()

	// Hash the token and save it
	hash := HashToken(token)
	SaveHash(string(hash))

	authenticationHash = hash
}

// package dashboard

// An Error is what the api replies when an error occurs
type Error struct {
	Message string `json:"message"`
}

// DeleteRequest is for the parameters required to delete an intent via the REST Api
type DeleteRequest struct {
	Tag string `json:"tag"`
}

// WriteIntents writes the given intents to the intents file
func WriteIntents(locale string, intents []Intent) {
	CacheIntents(locale, intents)

	// Encode the json
	bytes, _ := json.MarshalIndent(intents, "", "  ")

	// Write it to the file
	file, err := os.Create("res/locales/" + locale + "/intents.json")
	if err != nil {
		panic(err)
	}

	defer file.Close()

	file.Write(bytes)
}

// AddIntent adds the given intent to the intents file
func AddIntent(locale string, intent Intent) {
	intents := append(SerializeIntents(locale), intent)

	WriteIntents(locale, intents)

	fmt.Printf("Added %s intent.\n", color.FgMagenta.Render(intent.Tag))
}

// RemoveIntent removes the intent with the given tag from the intents file
func RemoveIntent(locale, tag string) {
	intents := SerializeIntents(locale)

	// Iterate through the intents to remove the right one
	for i, intent := range intents {
		if intent.Tag != tag {
			continue
		}

		intents[i] = intents[len(intents)-1]
		intents = intents[:len(intents)-1]
		fmt.Printf("The intent %s was deleted.\n", color.FgMagenta.Render(intent.Tag))
	}

	WriteIntents(locale, intents)
}

// GetIntents is the route to get the intents
func GetIntents(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")

	data := mux.Vars(r)

	// Encode the intents for the given locale
	json.NewEncoder(w).Encode(GetIntentsa(data["locale"]))
}

// CreateIntent is the route to create a new intent
func CreateIntent(w http.ResponseWriter, r *http.Request) {
	allowedHeaders := "Accept, Content-Type, Content-Length, Accept-Encoding, Authorization,Marboris-Token"
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
	w.Header().Set("Access-Control-Allow-Headers", allowedHeaders)
	w.Header().Set("Access-Control-Expose-Headers", "Authorization")

	data := mux.Vars(r)

	// Checks if the token present in the headers is the right one
	token := r.Header.Get("Marboris-Token")
	if !ChecksToken(token) {
		json.NewEncoder(w).Encode(Error{
			Message: GetMessageu(data["locale"], "no permission"),
		})
		return
	}

	// Decode request json body
	var intent Intent
	json.NewDecoder(r.Body).Decode(&intent)

	if intent.Responses == nil || intent.Patterns == nil {
		json.NewEncoder(w).Encode(Error{
			Message: GetMessageu(data["locale"], "patterns same"),
		})
		return
	}

	// Returns an error if the tags are the same
	for _, _intent := range GetIntentsa(data["locale"]) {
		if _intent.Tag == intent.Tag {
			json.NewEncoder(w).Encode(Error{
				Message: GetMessageu(data["locale"], "tags same"),
			})
			return
		}
	}

	// Adds the intent
	AddIntent(data["locale"], intent)

	json.NewEncoder(w).Encode(intent)
}

// DeleteIntent is the route used to delete an intent
func DeleteIntent(w http.ResponseWriter, r *http.Request) {
	allowedHeaders := "Accept, Content-Type, Content-Length, Accept-Encoding, Authorization,Marboris-Token"
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
	w.Header().Set("Access-Control-Allow-Headers", allowedHeaders)
	w.Header().Set("Access-Control-Expose-Headers", "Authorization")

	data := mux.Vars(r)

	// Checks if the token present in the headers is the right one
	token := r.Header.Get("Marboris-Token")
	if !ChecksToken(token) {
		json.NewEncoder(w).Encode(Error{
			Message: GetMessageu(data["locale"], "no permission"),
		})
		return
	}

	var deleteRequest DeleteRequest
	json.NewDecoder(r.Body).Decode(&deleteRequest)

	RemoveIntent(data["locale"], deleteRequest.Tag)

	json.NewEncoder(w).Encode(GetIntentsa(data["locale"]))
}

// package server

// Dashboard contains the data sent for the dashboard
type Dashboard struct {
	Layers   Layers   `json:"layers"`
	Training Training `json:"training"`
}

// Layers contains the data of the network's layers
type Layers struct {
	InputNodes   int `json:"input"`
	HiddenLayers int `json:"hidden"`
	OutputNodes  int `json:"output"`
}

// Training contains the data related to the training of the network
type Training struct {
	Rate   float64   `json:"rate"`
	Errors []float64 `json:"errors"`
	Time   float64   `json:"time"`
}

// GetDashboardData encodes the json for the dashboard data
func GetDashboardData(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")

	data := mux.Vars(r)

	dashboard := Dashboard{
		Layers:   GetLayers(data["locale"]),
		Training: GetTraining(data["locale"]),
	}

	err := json.NewEncoder(w).Encode(dashboard)
	if err != nil {
		log.Fatal(err)
	}
}

// GetLayers returns the number of input, hidden and output layers of the network
func GetLayers(locale string) Layers {
	return Layers{
		// Get the number of rows of the first layer to get the count of input nodes
		InputNodes: Rows(neuralNetworks[locale].Layers[0]),
		// Get the number of hidden layers by removing the count of the input and output layers
		HiddenLayers: len(neuralNetworks[locale].Layers) - 2,
		// Get the number of rows of the latest layer to get the count of output nodes
		OutputNodes: Columns(neuralNetworks[locale].Output),
	}
}

// GetTraining returns the learning rate, training date and error loss for the network
func GetTraining(locale string) Training {
	// Retrieve the information from the neural network
	return Training{
		Rate:   neuralNetworks[locale].Rate,
		Errors: neuralNetworks[locale].Errors,
		Time:   neuralNetworks[locale].Time,
	}
}

// package server

var (
	// Create the neural network variable to use it everywhere
	neuralNetworks map[string]Network
	// Initializes the cache with a 5 minute lifetime
	cache = gocache.New(5*time.Minute, 5*time.Minute)
)

// Serve serves the server in the given port
func Serve(_neuralNetworks map[string]Network, port string) {
	// Set the current global network as a global variable
	neuralNetworks = _neuralNetworks // require

	// Initializes the router
	router := mux.NewRouter()
	// Serve the websocket
	router.HandleFunc("/websocket", SocketHandle)
	// Serve the API
	router.HandleFunc("/api/{locale}/dashboard", GetDashboardData).Methods("GET")
	router.HandleFunc("/api/{locale}/intent", CreateIntent).Methods("POST")
	router.HandleFunc("/api/{locale}/intent", DeleteIntent).Methods("DELETE", "OPTIONS")
	router.HandleFunc("/api/{locale}/intents", GetIntents).Methods("GET")
	router.HandleFunc("/api/coverage", GetCoverage).Methods("GET")

	magenta := color.FgMagenta.Render
	fmt.Printf("\nServer listening on the port %s...\n", magenta(port))

	// Serves the chat
	err := http.ListenAndServe(":"+port, router)
	if err != nil {
		panic(err)
	}
}

// package server

// Configure the upgrader
var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

// RequestMessage is the structure that uses entry connections to chat with the websocket
type RequestMessage struct {
	Type        int         `json:"type"` // 0 for handshakes and 1 for messages
	Content     string      `json:"content"`
	Token       string      `json:"user_token"`
	Locale      string      `json:"locale"`
	Information Information `json:"information"`
}

// ResponseMessage is the structure used to reply to the user through the websocket
type ResponseMessage struct {
	Content     string      `json:"content"`
	Tag         string      `json:"tag"`
	Information Information `json:"information"`
}

// SocketHandle manages the entry connections and reply with the neural network
func SocketHandle(w http.ResponseWriter, r *http.Request) {
	conn, _ := upgrader.Upgrade(w, r, nil)
	fmt.Println(color.FgGreen.Render("A new connection has been opened"))

	for {
		// Read message from browser
		msgType, msg, err := conn.ReadMessage()
		if err != nil {
			break
		}

		// Unmarshal the json content of the message
		var request RequestMessage
		if err = json.Unmarshal(msg, &request); err != nil {
			continue
		}

		// Set the information from the client into the cache
		if reflect.DeepEqual(GetUserInformation(request.Token), Information{}) {
			SetUserInformation(request.Token, request.Information)
		}

		// If the type of requests is a handshake then execute the start modules
		if request.Type == 0 {
			ExecuteModules(request.Token, request.Locale)

			message := GetMessage()
			if message != "" {
				// Generate the response to send to the user
				response := ResponseMessage{
					Content:     message,
					Tag:         "start module",
					Information: GetUserInformation(request.Token),
				}

				bytes, err := json.Marshal(response)
				if err != nil {
					panic(err)
				}

				if err = conn.WriteMessage(msgType, bytes); err != nil {
					continue
				}
			}

			continue
		}

		// Write message back to browser
		response := Reply(request)
		if err = conn.WriteMessage(msgType, response); err != nil {
			continue
		}
	}
}

// Reply takes the entry message and returns an array of bytes for the answer
func Reply(request RequestMessage) []byte {
	var responseSentence, responseTag string

	// Send a message from res/datasets/messages.json if it is too long
	if len(request.Content) > 500 {
		responseTag = "too long"
		responseSentence = GetMessageu(request.Locale, responseTag)
	} else {
		// If the given locale is not supported yet, set english
		locale := request.Locale
		if !Exists(locale) {
			locale = "en"
		}

		responseTag, responseSentence = NewSentence(
			locale, request.Content,
		).Calculate(*cache, neuralNetworks[locale], request.Token)
	}

	// Marshall the response in json
	response := ResponseMessage{
		Content:     responseSentence,
		Tag:         responseTag,
		Information: GetUserInformation(request.Token),
	}

	bytes, err := json.Marshal(response)
	if err != nil {
		panic(err)
	}

	return bytes
}

// package en

func init() {
	RegisterModules("en", []Modulem{
		// AREA
		// For modules related to countries, please add the translations of the countries' names
		// or open an issue to ask for translations.

		{
			Tag: AreaTag,
			Patterns: []string{
				"What is the area of ",
				"Give me the area of ",
			},
			Responses: []string{
				"The area of %s is %gkm²",
			},
			Replacer: AreaReplacer,
		},

		// CAPITAL
		{
			Tag: CapitalTag,
			Patterns: []string{
				"What is the capital of ",
				"What's the capital of ",
				"Give me the capital of ",
			},
			Responses: []string{
				"The capital of %s is %s",
			},
			Replacer: CapitalReplacer,
		},

		// CURRENCY
		{
			Tag: CurrencyTag,
			Patterns: []string{
				"Which currency is used in ",
				"Give me the used currency of ",
				"Give me the currency of ",
				"What is the currency of ",
			},
			Responses: []string{
				"The currency of %s is %s",
			},
			Replacer: CurrencyReplacer,
		},

		// MATH
		// A regex translation is also required in `language/math.go`, please don't forget to translate it.
		// Otherwise, remove the registration of the Math module in this file.

		{
			Tag: MathTag,
			Patterns: []string{
				"Give me the result of ",
				"Calculate ",
			},
			Responses: []string{
				"The result is %s",
				"That makes %s",
			},
			Replacer: MathReplacer,
		},

		// MOVIES
		// A translation of movies genres is also required in `language/movies.go`, please don't forget
		// to translate it.
		// Otherwise, remove the registration of the Movies modules in this file.

		{
			Tag: GenresTag,
			Patterns: []string{
				"My favorite movie genres are Comedy, Horror",
				"I like the Comedy, Horror genres",
				"I like movies about War",
				"I like Action movies",
			},
			Responses: []string{
				"Great choices! I saved this movie genre information to your client.",
				"Understood, I saved this movie genre information to your client.",
			},
			Replacer: GenresReplacer,
		},

		{
			Tag: MoviesTag,
			Patterns: []string{
				"Find me a movie about",
				"Give me a movie about",
				"Find me a film about",
			},
			Responses: []string{
				"I found the movie “%s” for you, which is rated %.02f/5",
				"Sure, I found this movie “%s”, which is rated %.02f/5",
			},
			Replacer: MovieSearchReplacer,
		},

		{
			Tag: MoviesAlreadyTag,
			Patterns: []string{
				"I already saw this movie",
				"I have already watched this film",
				"Oh I have already watched this movie",
				"I have already seen this movie",
			},
			Responses: []string{
				"Oh I see, here's another one “%s” which is rated %.02f/5",
			},
			Replacer: MovieSearchReplacer,
		},

		{
			Tag: MoviesDataTag,
			Patterns: []string{
				"I'm bored",
				"I don't know what to do",
			},
			Responses: []string{
				"I propose you watch the %s movie “%s”, which is rated %.02f/5",
			},
			Replacer: MovieSearchFromInformationReplacer,
		},

		// NAME
		{
			Tag: NameGetterTag,
			Patterns: []string{
				"Do you know my name?",
			},
			Responses: []string{
				"Your name is %s!",
			},
			Replacer: NameGetterReplacer,
		},

		{
			Tag: NameSetterTag,
			Patterns: []string{
				"My name is ",
				"You can call me ",
			},
			Responses: []string{
				"Great! Hi %s",
			},
			Replacer: NameSetterReplacer,
		},

		// RANDOM
		{
			Tag: RandomTag,
			Patterns: []string{
				"Give me a random number",
				"Generate a random number",
			},
			Responses: []string{
				"The number is %s",
			},
			Replacer: RandomNumberReplacer,
		},

		{
			Tag: JokesTag,
			Patterns: []string{
				"Tell me a joke",
				"Make me laugh",
			},
			Responses: []string{
				"Here you go, %s",
				"Here's one, %s",
			},
			Replacer: JokesReplacer,
		},
		{
			Tag: AdvicesTag,
			Patterns: []string{
				"Give me an advice",
				"Advise me",
			},
			Responses: []string{
				"Here you go, %s",
				"Here's one, %s",
				"Listen closely, %s",
			},
			Replacer: AdvicesReplacer,
		},
	})

	// COUNTRIES
	// Please translate this method for adding the correct article in front of countries names.
	// Otherwise, remove the countries modules from this file.

	ArticleCountriesm["en"] = ArticleCountries
}

// ArticleCountries returns the country with its article in front.
func ArticleCountries(name string) string {
	if name == "United States" {
		return "the " + name
	}

	return name
}

// package language

// import (
// 	"encoding/json"
// 	"fmt"
// 	"strings"

// 	"marboris/core/cli"
// )

// Country is the serializer of the countries.json file in the res folder
type Country struct {
	Name     map[string]string `json:"name"`
	Capital  string            `json:"capital"`
	Code     string            `json:"code"`
	Area     float64           `json:"area"`
	Currency string            `json:"currency"`
}

var countries = SerializeCountries()

// SerializeCountries returns a list of countries, serialized from `res/datasets/countries.json`
func SerializeCountries() (countries []Country) {
	err := json.Unmarshal(ReadFile("res/datasets/countries.json"), &countries)
	if err != nil {
		fmt.Println(err)
	}

	return countries
}

// FindCountry returns the country found in the sentence and if no country is found, returns an empty Country struct
func FindCountry(locale, sentence string) Country {
	for _, country := range countries {
		name, exists := country.Name[locale]

		if !exists {
			continue
		}

		// If the actual country isn't contained in the sentence, continue
		if !strings.Contains(strings.ToLower(sentence), strings.ToLower(name)) {
			continue
		}

		// Returns the right country
		return country
	}

	// Returns an empty country if none has been found
	return Country{}
}

// package language

// Movie is the serializer from res/datasets/movies.csv
type Movie struct {
	Name   string
	Genres []string
	Rating float64
}

var (
	// MoviesGenres initializes movies genres in different languages
	MoviesGenres = map[string][]string{
		"en": {
			"Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
			"Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
		},
	}
	movies = SerializeMovies()
)

// SerializeMovies retrieves the content of res/datasets/movies.csv and serialize it
func SerializeMovies() (movies []Movie) {
	path := "res/datasets/movies.csv"
	bytes, err := os.Open(path)
	if err != nil {
		bytes, _ = os.Open("../" + path)
	}

	reader := csv.NewReader(bufio.NewReader(bytes))
	for {
		line, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			log.Fatal(err)
		}

		// Convert the string to a float
		rating, _ := strconv.ParseFloat(line[3], 64)

		movies = append(movies, Movie{
			Name:   line[1],
			Genres: strings.Split(line[2], "|"),
			Rating: rating,
		})
	}

	return
}

// SearchMovie search a movie for a given genre
func SearchMovie(genre, userToken string) (output Movie) {
	for _, movie := range movies {
		userMovieBlacklist := GetUserInformation(userToken).MovieBlacklist
		// Continue if the movie is not from the request genre or if this movie has already been suggested
		if !Contains(movie.Genres, genre) || Contains(userMovieBlacklist, movie.Name) {
			continue
		}

		if reflect.DeepEqual(output, Movie{}) || movie.Rating > output.Rating {
			output = movie
		}
	}

	// Add the found movie to the user blacklist
	ChangeUserInformation(userToken, func(information Information) Information {
		information.MovieBlacklist = append(information.MovieBlacklist, output.Name)
		return information
	})

	return
}

// FindMoviesGenres returns an array of genres found in the entry string
func FindMoviesGenres(locale, content string) (output []string) {
	for i, genre := range MoviesGenres[locale] {
		if LevenshteinContains(strings.ToUpper(content), strings.ToUpper(genre), 2) {
			output = append(output, MoviesGenres["en"][i])
		}
	}

	return
}

// package start

// A Module is a module that will be executed when a connection is opened by a user
type Modules struct {
	Action func(string, string)
}

var (
	moduless []Modules
	message  string
)

// RegisterModule registers the given module in the array
func RegisterModuless(module Modules) {
	moduless = append(moduless, module)
}

// SetMessage register the message which will be sent to the client
func SetMessage(_message string) {
	message = _message
}

// GetMessage returns the messages that needs to be sent
func GetMessage() string {
	return message
}

// ExecuteModules will execute all the registered start modules with the user token
func ExecuteModules(token, locale string) {
	fmt.Println(color.FgGreen.Render("Executing start modules.."))

	for _, module := range moduless {
		module.Action(token, locale)
	}
}

// package modules

const adviceURL = "https://api.adviceslip.com/advice"

// AdvicesTag is the intent tag for its module
var AdvicesTag = "advices"

// AdvicesReplacer replaces the pattern contained inside the response by a random advice from the api
// specified by the adviceURL.
// See modules/modules.go#Module.Replacer() for more details.
func AdvicesReplacer(locale, entry, response, _ string) (string, string) {

	resp, err := http.Get(adviceURL)
	if err != nil {
		responseTag := "no advices"
		return responseTag, GetMessageu(locale, responseTag)
	}

	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		responseTag := "no advices"
		return responseTag, GetMessageu(locale, responseTag)
	}

	var result map[string]interface{}
	json.Unmarshal(body, &result)
	advice := result["slip"].(map[string]interface{})["advice"]

	return AdvicesTag, fmt.Sprintf(response, advice)
}

// package modules

// AreaTag is the intent tag for its module
var AreaTag = "area"

// AreaReplacer replaces the pattern contained inside the response by the area of the country
// specified in the message.
// See modules/modules.go#Module.Replacer() for more details.
func AreaReplacer(locale, entry, response, _ string) (string, string) {
	country := FindCountry(locale, entry)

	// If there isn't a country respond with a message from res/datasets/messages.json
	if country.Currency == "" {
		responseTag := "no country"
		return responseTag, GetMessageu(locale, responseTag)
	}

	return AreaTag, fmt.Sprintf(response, ArticleCountriesm[locale](country.Name[locale]), country.Area)
}

// package modules

var (
	// CapitalTag is the intent tag for its module
	CapitalTag = "capital"
	// ArticleCountries is the map of functions to find the article in front of a country
	// in different languages
	ArticleCountriesm = map[string]func(string) string{}
)

// CapitalReplacer replaces the pattern contained inside the response by the capital of the country
// specified in the message.
// See modules/modules.go#Module.Replacer() for more details.
func CapitalReplacer(locale, entry, response, _ string) (string, string) {
	country := FindCountry(locale, entry)

	// If there isn't a country respond with a message from res/datasets/messages.json
	if country.Currency == "" {
		responseTag := "no country"
		return responseTag, GetMessageu(locale, responseTag)
	}

	articleFunction, exists := ArticleCountriesm[locale]
	countryName := country.Name[locale]
	if exists {
		countryName = articleFunction(countryName)
	}

	return CapitalTag, fmt.Sprintf(response, countryName, country.Capital)
}

// package modules

// CurrencyTag is the intent tag for its module
var CurrencyTag = "currency"

// CurrencyReplacer replaces the pattern contained inside the response by the currency of the country
// specified in the message.
// See modules/modules.go#Module.Replacer() for more details.
func CurrencyReplacer(locale, entry, response, _ string) (string, string) {
	country := FindCountry(locale, entry)

	// If there isn't a country respond with a message from res/datasets/messages.json
	if country.Currency == "" {
		responseTag := "no country"
		return responseTag, GetMessageu(locale, responseTag)
	}

	return CurrencyTag, fmt.Sprintf(response, ArticleCountriesm[locale](country.Name[locale]), country.Currency)
}

// package modules

const jokeURL = "https://official-joke-api.appspot.com/random_joke"

// JokesTag is the intent tag for its module
var JokesTag = "jokes"

// Joke represents the response from the joke api
type Joke struct {
	ID        int64  `json:"id"`
	Type      string `json:"type"`
	Setup     string `json:"setup"`
	Punchline string `json:"punchline"`
}

// JokesReplacer replaces the pattern contained inside the response by a random joke from the api
// specified in jokeURL.
// See modules/modules.go#Module.Replacer() for more details.
func JokesReplacer(locale, entry, response, _ string) (string, string) {

	resp, err := http.Get(jokeURL)
	if err != nil {
		responseTag := "no jokes"
		return responseTag, GetMessageu(locale, responseTag)
	}

	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		responseTag := "no jokes"
		return responseTag, GetMessageu(locale, responseTag)
	}

	joke := &Joke{}

	err = json.Unmarshal(body, joke)
	if err != nil {
		responseTag := "no jokes"
		return responseTag, GetMessageu(locale, responseTag)
	}

	jokeStr := joke.Setup + " " + joke.Punchline

	return JokesTag, fmt.Sprintf(response, jokeStr)
}

// package modules

// MathTag is the intent tag for its module
var MathTag = "math"

// MathReplacer replaces the pattern contained inside the response by the answer of the math
// expression specified in the message.
// See modules/modules.go#Module.Replacer() for more details.
func MathReplacer(locale, entry, response, _ string) (string, string) {
	operation := FindMathOperation(entry)

	// If there is no operation in the entry message reply with a "don't understand" message
	if operation == "" {
		responseTag := "don't understand"
		return responseTag, GetMessageu(locale, responseTag)
	}

	res, err := mathcat.Eval(operation)
	// If the expression isn't valid reply with a message from res/datasets/messages.json
	if err != nil {
		responseTag := "math not valid"
		return responseTag, GetMessageu(locale, responseTag)
	}
	// Use number of decimals from the query
	decimals := FindNumberOfDecimals(locale, entry)
	if decimals == 0 {
		decimals = 6
	}

	result := res.FloatString(decimals)

	// Remove trailing zeros of the result with a Regex
	trailingZerosRegex := regexp.MustCompile(`\.?0+$`)
	result = trailingZerosRegex.ReplaceAllString(result, "")

	return MathTag, fmt.Sprintf(response, result)
}

// package modules

// Module is a structure for dynamic intents with a Tag, some Patterns and Responses and
// a Replacer function to execute the dynamic changes.
type Modulem struct {
	Tag       string
	Patterns  []string
	Responses []string
	Replacer  func(string, string, string, string) (string, string)
	Context   string
}

var modulesm = map[string][]Modulem{}

// RegisterModule registers a module into the map
func RegisterModule(locale string, module Modulem) {
	modulesm[locale] = append(modulesm[locale], module)
}

// RegisterModules registers an array of modulesm into the map
func RegisterModules(locale string, _modules []Modulem) {
	modulesm[locale] = append(modulesm[locale], _modules...)
}

// GetModules returns all the registered modulesm
func GetModules(locale string) []Modulem {
	return modulesm[locale]
}

// GetModuleByTag returns a module found by the given tag and locale
func GetModuleByTag(tag, locale string) Modulem {
	for _, module := range modulesm[locale] {
		if tag != module.Tag {
			continue
		}

		return module
	}

	return Modulem{}
}

// ReplaceContent apply the Replacer of the matching module to the response and returns it
func ReplaceContent(locale, tag, entry, response, token string) (string, string) {
	for _, module := range modulesm[locale] {
		if module.Tag != tag {
			continue
		}

		return module.Replacer(locale, entry, response, token)
	}

	return tag, response
}

// package modules

var (
	// GenresTag is the intent tag for its module
	GenresTag = "movies genres"
	// MoviesTag is the intent tag for its module
	MoviesTag = "movies search"
	// MoviesAlreadyTag is the intent tag for its module
	MoviesAlreadyTag = "already seen movie"
	// MoviesDataTag is the intent tag for its module
	MoviesDataTag = "movies search from data"
)

// GenresReplacer gets the genre specified in the message and adds it to the user information.
// See modules/modules.go#Module.Replacer() for more details.
func GenresReplacer(locale, entry, response, token string) (string, string) {
	genres := FindMoviesGenres(locale, entry)

	// If there is no genres then reply with a message from res/datasets/messages.json
	if len(genres) == 0 {
		responseTag := "no genres"
		return responseTag, GetMessageu(locale, responseTag)
	}

	// Change the user information to add the new genres
	ChangeUserInformation(token, func(information Information) Information {
		for _, genre := range genres {
			// Append the genre only is it isn't already in the information
			if Contains(information.MovieGenres, genre) {
				continue
			}

			information.MovieGenres = append(information.MovieGenres, genre)
		}
		return information
	})

	return GenresTag, response
}

// MovieSearchReplacer replaces the patterns contained inside the response by the movie's name
// and rating from the genre specified in the message.
// See modules/modules.go#Module.Replacer() for more details.
func MovieSearchReplacer(locale, entry, response, token string) (string, string) {
	genres := FindMoviesGenres(locale, entry)

	// If there is no genres then reply with a message from res/datasets/messages.json
	if len(genres) == 0 {
		responseTag := "no genres"
		return responseTag, GetMessageu(locale, responseTag)
	}

	movie := SearchMovie(genres[0], token)

	return MoviesTag, fmt.Sprintf(response, movie.Name, movie.Rating)
}

// MovieSearchFromInformationReplacer replaces the patterns contained inside the response by the movie's name
// and rating from the genre in the user's information.
// See modules/modules.go#Module.Replacer() for more details.
func MovieSearchFromInformationReplacer(locale, _, response, token string) (string, string) {
	// If there is no genres then reply with a message from res/datasets/messages.json
	genres := GetUserInformation(token).MovieGenres
	if len(genres) == 0 {
		responseTag := "no genres saved"
		return responseTag, GetMessageu(locale, responseTag)
	}

	movie := SearchMovie(genres[rand.Intn(len(genres))], token)
	genresJoined := strings.Join(genres, ", ")
	return MoviesDataTag, fmt.Sprintf(response, genresJoined, movie.Name, movie.Rating)
}

// package modules

var (
	// NameGetterTag is the intent tag for its module
	NameGetterTag = "name getter"
	// NameSetterTag is the intent tag for its module
	NameSetterTag = "name setter"
)

// NameGetterReplacer replaces the pattern contained inside the response by the user's name.
// See modules/modules.go#Module.Replacer() for more details.
func NameGetterReplacer(locale, _, response, token string) (string, string) {
	name := GetUserInformation(token).Name

	if strings.TrimSpace(name) == "" {
		responseTag := "don't know name"
		return responseTag, GetMessageu(locale, responseTag)
	}

	return NameGetterTag, fmt.Sprintf(response, name)
}

// NameSetterReplacer gets the name specified in the message and save it in the user's information.
// See modules/modules.go#Module.Replacer() for more details.
func NameSetterReplacer(locale, entry, response, token string) (string, string) {
	name := FindName(entry)

	// If there is no name in the entry string
	if name == "" {
		responseTag := "no name"
		return responseTag, GetMessageu(locale, responseTag)
	}

	// Capitalize the name
	name = strings.Title(name)

	// Change the name inside the user information
	ChangeUserInformation(token, func(information Information) Information {
		information.Name = name
		return information
	})

	return NameSetterTag, fmt.Sprintf(response, name)
}

// package modules

// RandomTag is the intent tag for its module
var RandomTag = "random number"

// RandomNumberReplacer replaces the pattern contained inside the response by a random number.
// See modules/modules.go#Module.Replacer() for more details.
func RandomNumberReplacer(locale, entry, response, _ string) (string, string) {
	limitArr, err := FindRangeLimits(locale, entry)
	if err != nil {
		if limitArr != nil {
			return RandomTag, fmt.Sprintf(response, strconv.Itoa(rand.Intn(100)))
		}

		responseTag := "no random range"
		return responseTag, GetMessageu(locale, responseTag)
	}

	min := limitArr[0]
	max := limitArr[1]
	randNum := rand.Intn((max - min)) + min
	return RandomTag, fmt.Sprintf(response, strconv.Itoa(randNum))
}

// package date

// PatternTranslation are the map of regexs in different languages
var PatternTranslation = map[string]PatternTranslations{
	"en": {
		DateRegex: `(of )?(the )?((after )?tomorrow|((today|tonight)|(next )?(monday|tuesday|wednesday|thursday|friday|saturday|sunday))|(\d{2}|\d)(th|rd|st|nd)? (of )?(january|february|march|april|may|june|july|august|september|october|november|december)|((\d{2}|\d)/(\d{2}|\d)))`,
		TimeRegex: `(at )?(\d{2}|\d)(:\d{2}|\d)?( )?(pm|am|p\.m|a\.m)`,
	},
}

// PatternTranslations are the translations of the regexs for dates
type PatternTranslations struct {
	DateRegex string
	TimeRegex string
}

// SearchTime returns the found date in the given sentence and the sentence without the date, if no date has
// been found, it returns an empty date and the given sentence.
func SearchTime(locale, sentence string) (string, time.Time) {
	_time := RuleTime(sentence)
	// Set the time to 12am if no time has been found
	if _time == (time.Time{}) {
		_time = time.Date(0, 0, 0, 12, 0, 0, 0, time.UTC)
	}

	for _, rule := range rules {
		date := rule(locale, sentence)

		// If the current rule found a date
		if date != (time.Time{}) {
			date = time.Date(date.Year(), date.Month(), date.Day(), _time.Hour(), _time.Minute(), 0, 0, time.UTC)

			sentence = DeleteTimes(locale, sentence)
			return DeleteDates(locale, sentence), date
		}
	}

	return sentence, time.Now().Add(time.Hour * 24)
}

// DeleteDates removes the dates of the given sentence and returns it
func DeleteDates(locale, sentence string) string {
	// Create a regex to match the patterns of dates to remove them.
	datePatterns := regexp.MustCompile(PatternTranslation[locale].DateRegex)

	// Replace the dates by empty string
	sentence = datePatterns.ReplaceAllString(sentence, "")
	// Trim the spaces and return
	return strings.TrimSpace(sentence)
}

// DeleteTimes removes the times of the given sentence and returns it
func DeleteTimes(locale, sentence string) string {
	// Create a regex to match the patterns of times to remove them.
	timePatterns := regexp.MustCompile(PatternTranslation[locale].TimeRegex)

	// Replace the times by empty string
	sentence = timePatterns.ReplaceAllString(sentence, "")
	// Trim the spaces and return
	return strings.TrimSpace(sentence)
}

// package date

// A Rule is a function that takes the given sentence and tries to parse a specific
// rule to return a date, if not, the date is empty.
type Rule func(string, string) time.Time

var rules []Rule

// RegisterRule takes a rule in parameter and register it to the array of rules
func RegisterRule(rule Rule) {
	rules = append(rules, rule)
}

// package date

const day = time.Hour * 24

// RuleTranslations are the translations of the rules in different languages
var RuleTranslations = map[string]RuleTranslation{
	"en": {
		DaysOfWeek: []string{
			"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
		},
		Months: []string{
			"january", "february", "march", "april", "may", "june", "july",
			"august", "september", "october", "november", "december",
		},
		RuleToday:         `today|tonight`,
		RuleTomorrow:      `(after )?tomorrow`,
		RuleAfterTomorrow: "after",
		RuleDayOfWeek:     `(next )?(monday|tuesday|wednesday|thursday|friday|saturday|sunday)`,
		RuleNextDayOfWeek: "next",
		RuleNaturalDate:   `january|february|march|april|may|june|july|august|september|october|november|december`,
	},
}

// A RuleTranslation is all the texts/regexs to match the dates
type RuleTranslation struct {
	DaysOfWeek        []string
	Months            []string
	RuleToday         string
	RuleTomorrow      string
	RuleAfterTomorrow string
	RuleDayOfWeek     string
	RuleNextDayOfWeek string
	RuleNaturalDate   string
}

var daysOfWeek = map[string]time.Weekday{
	"monday":    time.Monday,
	"tuesday":   time.Tuesday,
	"wednesday": time.Wednesday,
	"thursday":  time.Thursday,
	"friday":    time.Friday,
	"saturday":  time.Saturday,
	"sunday":    time.Sunday,
}

func init() {
	// Register the rules
	RegisterRule(RuleToday)
	RegisterRule(RuleTomorrow)
	RegisterRule(RuleDayOfWeek)
	RegisterRule(RuleNaturalDate)
	RegisterRule(RuleDate)
}

// RuleToday checks for today, tonight, this afternoon dates in the given sentence, then
// it returns the date parsed.
func RuleToday(locale, sentence string) (result time.Time) {
	todayRegex := regexp.MustCompile(RuleTranslations[locale].RuleToday)
	today := todayRegex.FindString(sentence)

	// Returns an empty date struct if no date has been found
	if today == "" {
		return time.Time{}
	}

	return time.Now()
}

// RuleTomorrow checks for "tomorrow" and "after tomorrow" dates in the given sentence, then
// it returns the date parsed.
func RuleTomorrow(locale, sentence string) (result time.Time) {
	tomorrowRegex := regexp.MustCompile(RuleTranslations[locale].RuleTomorrow)
	date := tomorrowRegex.FindString(sentence)

	// Returns an empty date struct if no date has been found
	if date == "" {
		return time.Time{}
	}

	result = time.Now().Add(day)

	// If the date contains "after", we add 24 hours to tomorrow's date
	if strings.Contains(date, RuleTranslations[locale].RuleAfterTomorrow) {
		return result.Add(day)
	}

	return
}

// RuleDayOfWeek checks for the days of the week and the keyword "next" in the given sentence,
// then it returns the date parsed.
func RuleDayOfWeek(locale, sentence string) time.Time {
	dayOfWeekRegex := regexp.MustCompile(RuleTranslations[locale].RuleDayOfWeek)
	date := dayOfWeekRegex.FindString(sentence)

	// Returns an empty date struct if no date has been found
	if date == "" {
		return time.Time{}
	}

	var foundDayOfWeek int
	// Find the integer value of the found day of the week
	for _, dayOfWeek := range daysOfWeek {
		// Down case the day of the week to match the found date
		stringDayOfWeek := strings.ToLower(dayOfWeek.String())

		if strings.Contains(date, stringDayOfWeek) {
			foundDayOfWeek = int(dayOfWeek)
		}
	}

	currentDay := int(time.Now().Weekday())
	// Calculate the date of the found day
	calculatedDate := foundDayOfWeek - currentDay

	// If the day is already passed in the current week, then we add another week to the count
	if calculatedDate <= 0 {
		calculatedDate += 7
	}

	// If there is "next" in the sentence, then we add another week
	if strings.Contains(date, RuleTranslations[locale].RuleNextDayOfWeek) {
		calculatedDate += 7
	}

	// Then add the calculated number of day to the actual date
	return time.Now().Add(day * time.Duration(calculatedDate))
}

// RuleNaturalDate checks for the dates written in natural language in the given sentence,
// then it returns the date parsed.
func RuleNaturalDate(locale, sentence string) time.Time {
	naturalMonthRegex := regexp.MustCompile(
		RuleTranslations[locale].RuleNaturalDate,
	)
	naturalDayRegex := regexp.MustCompile(`\d{2}|\d`)

	month := naturalMonthRegex.FindString(sentence)
	day := naturalDayRegex.FindString(sentence)

	// Put the month in english to parse the time with time golang package
	if locale != "en" {
		monthIndex := Index(RuleTranslations[locale].Months, month)
		month = RuleTranslations["en"].Months[monthIndex]
	}

	parsedMonth, _ := time.Parse("January", month)
	parsedDay, _ := strconv.Atoi(day)

	// Returns an empty date struct if no date has been found
	if day == "" && month == "" {
		return time.Time{}
	}

	// If only the month is specified
	if day == "" {
		// Calculate the number of months to add
		calculatedMonth := parsedMonth.Month() - time.Now().Month()
		// Add a year if the month is passed
		if calculatedMonth <= 0 {
			calculatedMonth += 12
		}

		// Remove the number of days elapsed in the month to reach the first
		return time.Now().AddDate(0, int(calculatedMonth), -time.Now().Day()+1)
	}

	// Parse the date
	parsedDate := fmt.Sprintf("%d-%02d-%02d", time.Now().Year(), parsedMonth.Month(), parsedDay)
	date, err := time.Parse("2006-01-02", parsedDate)
	if err != nil {
		return time.Time{}
	}

	// If the date has been passed, add a year
	if time.Now().After(date) {
		date = date.AddDate(1, 0, 0)
	}

	return date
}

// RuleDate checks for dates written like mm/dd
func RuleDate(locale, sentence string) time.Time {
	dateRegex := regexp.MustCompile(`(\d{2}|\d)/(\d{2}|\d)`)
	date := dateRegex.FindString(sentence)

	// Returns an empty date struct if no date has been found
	if date == "" {
		return time.Time{}
	}

	// Parse the found date
	parsedDate, err := time.Parse("01/02", date)
	if err != nil {
		return time.Time{}
	}

	// Add the current year to the date
	parsedDate = parsedDate.AddDate(time.Now().Year(), 0, 0)

	// Add another year if the date is passed
	if time.Now().After(parsedDate) {
		parsedDate = parsedDate.AddDate(1, 0, 0)
	}

	return parsedDate
}

// RuleTime checks for an hour written like 9pm
func RuleTime(sentence string) time.Time {
	timeRegex := regexp.MustCompile(`(\d{2}|\d)(:\d{2}|\d)?( )?(pm|am|p\.m|a\.m)`)
	foundTime := timeRegex.FindString(sentence)

	// Returns an empty date struct if no date has been found
	if foundTime == "" {
		return time.Time{}
	}

	// Initialize the part of the day asked
	var part string
	if strings.Contains(foundTime, "pm") || strings.Contains(foundTime, "p.m") {
		part = "pm"
	} else if strings.Contains(foundTime, "am") || strings.Contains(foundTime, "a.m") {
		part = "am"
	}

	if strings.Contains(foundTime, ":") {
		// Get the hours and minutes of the given time
		hoursAndMinutesRegex := regexp.MustCompile(`(\d{2}|\d):(\d{2}|\d)`)
		timeVariables := strings.Split(hoursAndMinutesRegex.FindString(foundTime), ":")

		// Format the time with 2 digits for each
		formattedTime := fmt.Sprintf("%02s:%02s %s", timeVariables[0], timeVariables[1], part)
		response, _ := time.Parse("03:04 pm", formattedTime)

		return response
	}

	digitsRegex := regexp.MustCompile(`\d{2}|\d`)
	foundDigits := digitsRegex.FindString(foundTime)

	formattedTime := fmt.Sprintf("%02s %s", foundDigits, part)
	response, _ := time.Parse("03 pm", formattedTime)

	return response
}

// package language

// LevenshteinDistance calculates the Levenshtein Distance between two given words and returns it.
// Please see https://en.wikipedia.org/wiki/Levenshtein_distance.
func LevenshteinDistance(first, second string) int {
	// Returns the length if it's empty
	if first == "" {
		return len(second)
	}
	if second == "" {
		return len(first)
	}

	if first[0] == second[0] {
		return LevenshteinDistance(first[1:], second[1:])
	}

	a := LevenshteinDistance(first[1:], second[1:])
	if b := LevenshteinDistance(first, second[1:]); a > b {
		a = b
	}

	if c := LevenshteinDistance(first[1:], second); a > c {
		a = c
	}

	return a + 1
}

// LevenshteinContains checks for a given matching string in a given sentence with a minimum rate for Levenshtein.
func LevenshteinContains(sentence, matching string, rate int) bool {
	words := strings.Split(sentence, " ")
	for _, word := range words {
		// Returns true if the distance is below the rate
		if LevenshteinDistance(word, matching) <= rate {
			return true
		}
	}

	return false
}

// package language

// MathDecimals is the map for having the regex on decimals in different languages
var MathDecimals = map[string]string{
	"en": `(\d+( |-)decimal(s)?)|(number (of )?decimal(s)? (is )?\d+)`,
}

// FindMathOperation finds a math operation in a string an returns it
func FindMathOperation(entry string) string {
	mathRegex := regexp.MustCompile(
		`((\()?(((\d+|pi)(\^\d+|!|.)?)|sqrt|cos|sin|tan|acos|asin|atan|log|ln|abs)( )?[+*\/\-x]?( )?(\))?[+*\/\-]?)+`,
	)

	operation := mathRegex.FindString(entry)
	// Replace "x" symbol by "*"
	operation = strings.Replace(operation, "x", "*", -1)
	return strings.TrimSpace(operation)
}

// FindNumberOfDecimals finds the number of decimals asked in the query
func FindNumberOfDecimals(locale, entry string) int {
	decimalsRegex := regexp.MustCompile(
		MathDecimals[locale],
	)
	numberRegex := regexp.MustCompile(`\d+`)

	decimals := numberRegex.FindString(decimalsRegex.FindString(entry))
	decimalsInt, _ := strconv.Atoi(decimals)

	return decimalsInt
}

// package language

var names = SerializeNames()

// SerializeNames retrieves all the names from res/datasets/names.txt and returns an array of names
func SerializeNames() (names []string) {
	namesFile := string(ReadFile("res/datasets/names.txt"))

	// Iterate each line of the file
	names = append(names, strings.Split(strings.TrimSuffix(namesFile, "\n"), "\n")...)
	return
}

// FindName returns a name found in the given sentence or an empty string if no name has been found
func FindName(sentence string) string {
	for _, name := range names {
		if !strings.Contains(strings.ToLower(" "+sentence+" "), " "+name+" ") {
			continue
		}

		return name
	}

	return ""
}

// package language

var decimal = "\\b\\d+([\\.,]\\d+)?"

// FindRangeLimits finds the range for random numbers and returns a sorted integer array
func FindRangeLimits(local, entry string) ([]int, error) {
	decimalsRegex := regexp.MustCompile(decimal)
	limitStrArr := decimalsRegex.FindAllString(entry, 2)
	limitArr := make([]int, 0)

	if limitStrArr == nil {
		return make([]int, 0), errors.New("no range")
	}

	if len(limitStrArr) != 2 {
		return nil, errors.New("need 2 numbers, a lower and upper limit")
	}

	for _, v := range limitStrArr {
		num, err := strconv.Atoi(v)
		if err != nil {
			return nil, errors.New("non integer range")
		}
		limitArr = append(limitArr, num)
	}

	sort.Ints(limitArr)
	return limitArr, nil
}

// package language

// SearchTokens searches 2 tokens in the given sentence and returns it.
func SearchTokens(sentence string) []string {
	// Search the token with a regex
	tokenRegex := regexp.MustCompile(`[a-z0-9]{32}`)
	// Returns the found token
	return tokenRegex.FindAllString(sentence, 2)
}
