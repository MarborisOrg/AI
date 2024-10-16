package core

import (
	"bufio"
	crand "crypto/rand"
	"encoding/csv"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand"
	"net/http"
	"os"
	"os/user"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gookit/color"

	"github.com/soudy/mathcat"

	"golang.org/x/crypto/bcrypt"
	"gopkg.in/cheggaaa/pb.v1"

	"github.com/gorilla/mux"
	"github.com/tebeka/snowball"

	gocache "github.com/patrickmn/go-cache"
)

func Contains(slice []string, text string) bool {
	for _, item := range slice {
		if item == text {
			return true
		}
	}

	return false
}

func Difference(slice []string, slice2 []string) (difference []string) {
	for i := 0; i < 2; i++ {
		for _, s1 := range slice {
			found := false
			for _, s2 := range slice2 {
				if s1 == s2 {
					found = true
					break
				}
			}

			if !found {
				difference = append(difference, s1)
			}
		}

		if i == 0 {
			slice, slice2 = slice2, slice
		}
	}

	return difference
}

func Index(slice []string, text string) int {
	for i, item := range slice {
		if item == text {
			return i
		}
	}

	return 0
}

func SerializeMessages(locale string) []Message {
	var currentMessages []Message
	err := json.Unmarshal(ReadFile(GetResDir("locales", "messages.json", locale)), &currentMessages)
	if err != nil {
		fmt.Println(err)
	}

	messages[locale] = currentMessages

	return currentMessages
}

func GetMessages(locale string) []Message {
	return messages[locale]
}

func GetMessageByTag(tag, locale string) Message {
	for _, message := range messages[locale] {
		if tag != message.Tag {
			continue
		}

		return message
	}

	return Message{}
}

func GetMessageu(locale, tag string) string {
	for _, message := range messages[locale] {

		if message.Tag != tag {
			continue
		}

		if len(message.Messages) == 1 {
			return message.Messages[0]
		}

		rand.NewSource(time.Now().UnixNano()) // Seed
		return message.Messages[rand.Intn(len(message.Messages))]
	}

	return ""
}

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

func ChangeUserInformation(token string, changer func(Information) Information) {
	userInformation[token] = changer(userInformation[token])
}

func SetUserInformation(token string, information Information) {
	userInformation[token] = information
}

func GetUserInformation(token string) Information {
	return userInformation[token]
}

func getTempDir(fileName string) (filePath string) {
	tempDir := os.TempDir()
	return filepath.Join(tempDir, fileName)
}

func CreateNeuralNetwork(locale string) (neuralNetwork Network) {
	saveFile := getTempDir("Marboris-Training.json")

	_, err := os.Open(saveFile)

	if err != nil {
		panic("No training data found.")
	} else {
		fmt.Printf(
			"%s %s\n",
			color.FgBlue.Render("Loading the neural network from"),
			color.FgRed.Render(saveFile),
		)

		SerializeIntents(locale)
		neuralNetwork = *LoadNetwork(saveFile)
	}

	return
}

func (network Network) ComputeLastLayerDerivatives() Derivative {
	l := len(network.Layers) - 1
	lastLayer := network.Layers[l]

	cost := Differencen(network.Output, lastLayer)
	sigmoidDerivative := Multiplication(lastLayer, ApplyFunction(lastLayer, SubtractsOne))

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

func (network Network) ComputeDerivatives(i int, derivatives []Derivative) Derivative {
	l := len(network.Layers) - 2 - i

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

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func MultipliesByTwo(x float64) float64 {
	return 2 * x
}

func SubtractsOne(x float64) float64 {
	return x - 1
}

func CreateMatrix(rows, columns int) (matrix Matrix) {
	matrix = make(Matrix, rows)

	for i := 0; i < rows; i++ {
		matrix[i] = make([]float64, columns)
	}

	return
}

func Rows(matrix Matrix) int {
	return len(matrix)
}

func Columns(matrix Matrix) int {
	return len(matrix[0])
}

func ApplyFunctionWithIndex(matrix Matrix, fn func(i, j int, x float64) float64) Matrix {
	for i := 0; i < Rows(matrix); i++ {
		for j := 0; j < Columns(matrix); j++ {
			matrix[i][j] = fn(i, j, matrix[i][j])
		}
	}

	return matrix
}

func ApplyFunction(matrix Matrix, fn func(x float64) float64) Matrix {
	return ApplyFunctionWithIndex(matrix, func(i, j int, x float64) float64 {
		return fn(x)
	})
}

func ApplyRate(matrix Matrix, rate float64) Matrix {
	return ApplyFunction(matrix, func(x float64) float64 {
		return rate * x
	})
}

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

func Sum(matrix, matrix2 Matrix) (resultMatrix Matrix) {
	ErrorNotSameSize(matrix, matrix2)

	resultMatrix = CreateMatrix(Rows(matrix), Columns(matrix)) // warning is fine

	return ApplyFunctionWithIndex(matrix, func(i, j int, x float64) float64 {
		return matrix[i][j] + matrix2[i][j]
	})
}

func Differencen(matrix, matrix2 Matrix) (resultMatrix Matrix) {
	ErrorNotSameSize(matrix, matrix2)

	resultMatrix = CreateMatrix(Rows(matrix), Columns(matrix))

	return ApplyFunctionWithIndex(resultMatrix, func(i, j int, x float64) float64 {
		return matrix[i][j] - matrix2[i][j]
	})
}

func Multiplication(matrix, matrix2 Matrix) (resultMatrix Matrix) {
	ErrorNotSameSize(matrix, matrix2)

	resultMatrix = CreateMatrix(Rows(matrix), Columns(matrix)) // warning is fine

	return ApplyFunctionWithIndex(matrix, func(i, j int, x float64) float64 {
		return matrix[i][j] * matrix2[i][j]
	})
}

func Transpose(matrix Matrix) (resultMatrix Matrix) {
	resultMatrix = CreateMatrix(Columns(matrix), Rows(matrix))

	for i := 0; i < Rows(matrix); i++ {
		for j := 0; j < Columns(matrix); j++ {
			resultMatrix[j][i] = matrix[i][j]
		}
	}

	return resultMatrix
}

func ErrorNotSameSize(matrix, matrix2 Matrix) {
	if Rows(matrix) != Rows(matrix2) && Columns(matrix) != Columns(matrix2) {
		panic("These two matrices must have the same dimension.")
	}
}

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

func (network Network) Save(fileName string) {
	outF, err := os.OpenFile(fileName, os.O_CREATE|os.O_RDWR, 0o777)
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

func (network *Network) FeedForward() {
	for i := 0; i < len(network.Layers)-1; i++ {
		layer, weights, biases := network.Layers[i], network.Weights[i], network.Biases[i]

		productMatrix := DotProduct(layer, weights)
		Sum(productMatrix, biases)
		ApplyFunction(productMatrix, Sigmoid)

		network.Layers[i+1] = productMatrix
	}
}

func (network *Network) Predict(input []float64) []float64 {
	network.Layers[0] = Matrix{input}
	network.FeedForward()
	return network.Layers[len(network.Layers)-1][0]
}

func (network *Network) FeedBackward() {
	var derivatives []Derivative
	derivatives = append(derivatives, network.ComputeLastLayerDerivatives())

	for i := 0; i < len(network.Layers)-2; i++ {
		derivatives = append(derivatives, network.ComputeDerivatives(i, derivatives))
	}

	network.Adjust(derivatives)
}

func (network *Network) ComputeError() float64 {
	network.FeedForward()
	lastLayer := network.Layers[len(network.Layers)-1]
	errors := Differencen(network.Output, lastLayer)

	var i int
	var sum float64
	for _, a := range errors {
		for _, e := range a {
			sum += e
			i++
		}
	}

	return sum / float64(i)
}

func (network *Network) Train(iterations int) {
	start := time.Now()

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

	for i := 0; i < iterations; i++ {
		network.FeedForward()
		network.FeedBackward()

		if i%(iterations/20) == 0 {
			network.Errors = append(
				network.Errors,

				network.ComputeError(),
			)
		}

		bar.Increment()
	}

	bar.Finish()

	arrangedError := fmt.Sprintf("%.5f", network.ComputeError())

	elapsed := time.Since(start)

	network.Time = math.Floor(elapsed.Seconds()*100) / 100

	fmt.Printf("The error rate is %s.\n", color.FgGreen.Render(arrangedError))
}

func NewSentence(locale, content string) (sentence Sentence) {
	sentence = Sentence{
		Locale:  locale,
		Content: content,
	}
	sentence.arrange()

	return
}

func (sentence Sentence) PredictTag(neuralNetwork Network) string {
	words, classes, _ := Organize(sentence.Locale)

	predict := neuralNetwork.Predict(sentence.WordsBag(words))

	var resultsTag []Result
	for i, result := range predict {
		if i >= len(classes) {
			continue
		}
		resultsTag = append(resultsTag, Result{classes[i], result})
	}

	sort.Slice(resultsTag, func(i, j int) bool {
		return resultsTag[i].Value > resultsTag[j].Value
	})

	LogResults(sentence.Locale, sentence.Content, resultsTag)

	return resultsTag[0].Tag
}

func RandomizeResponse(locale, entry, tag, token string) (string, string) {
	if tag == DontUnderstand {
		return DontUnderstand, GetMessageu(locale, tag)
	}

	intents := append(SerializeIntents(locale), SerializeModulesIntents(locale)...)

	for _, intent := range intents {
		if intent.Tag != tag {
			continue
		}

		cacheTag, _ := userCache.Get(token)
		if intent.Context != "" && cacheTag != intent.Context {
			return DontUnderstand, GetMessageu(locale, DontUnderstand)
		}

		userCache.Set(token, tag, gocache.DefaultExpiration)

		response := intent.Responses[0]
		if len(intent.Responses) > 1 {
			rand.New(rand.NewSource(time.Now().UnixNano()))
			response = intent.Responses[rand.Intn(len(intent.Responses))]
		}

		return ReplaceContent(locale, tag, entry, response, token)
	}

	return DontUnderstand, GetMessageu(locale, DontUnderstand)
}

func (sentence Sentence) Calculate(cache gocache.Cache, neuralNetwork Network, token string) (string, string) {
	tag, found := cache.Get(sentence.Content)

	if !found {
		tag = sentence.PredictTag(neuralNetwork)
		cache.Set(sentence.Content, tag, gocache.DefaultExpiration)
	}

	return RandomizeResponse(sentence.Locale, sentence.Content, tag.(string), token)
}

func LogResults(locale, entry string, results []Result) {
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

		if result.Value < 0.004 {
			continue
		}

		fmt.Printf("  %s %s - %s\n", green("▫︎"), result.Tag, yellow(result.Value))
	}
}

func CacheIntents(locale string, _intents []Intent) {
	intents[locale] = _intents
}

func GetIntentsa(locale string) []Intent {
	return intents[locale]
}

func GetResDir(dir string, file string, dir2 ...string) (filePath string) {
	usr, err := user.Current()
	if err != nil {
		fmt.Println("Error getting user home directory:", err)
		panic("[Training] No user.. (419)")
	}
	homeDir := usr.HomeDir

	if dir == "" {
		return filepath.Join(homeDir, ".marboris", "res", file)
	}

	if len(dir2) == 0 || dir2[0] == "" {
		return filepath.Join(homeDir, ".marboris", "res", dir, file)
	}

	return filepath.Join(homeDir, ".marboris", "res", dir, dir2[0], file)
}

func SerializeIntents(locale string) (_intents []Intent) {
	err := json.Unmarshal(ReadFile(GetResDir("locales", "intents.json", locale)), &_intents)
	if err != nil {
		panic(err)
	}

	CacheIntents(locale, _intents)

	return _intents
}

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

func GetIntentByTag(tag, locale string) Intent {
	for _, intent := range GetIntentsa(locale) {
		if tag != intent.Tag {
			continue
		}

		return intent
	}

	return Intent{}
}

func Organize(locale string) (words, classes []string, documents []Document) {
	intents := append(
		SerializeIntents(locale),
		SerializeModulesIntents(locale)...,
	)

	for _, intent := range intents {
		for _, pattern := range intent.Patterns {

			patternSentence := Sentence{locale, pattern}
			patternSentence.arrange()

			for _, word := range patternSentence.stem() {
				if !Contains(words, word) {
					words = append(words, word)
				}
			}

			documents = append(documents, Document{
				patternSentence,
				intent.Tag,
			})
		}

		classes = append(classes, intent.Tag)
	}

	sort.Strings(words)
	sort.Strings(classes)

	return words, classes, documents
}

func (sentence *Sentence) arrange() {
	punctuationRegex := regexp.MustCompile(`[a-zA-Z]( )?(\.|\?|!|¿|¡)`)
	sentence.Content = punctuationRegex.ReplaceAllStringFunc(sentence.Content, func(s string) string {
		punctuation := regexp.MustCompile(`(\.|\?|!)`)
		return punctuation.ReplaceAllString(s, "")
	})

	sentence.Content = strings.ReplaceAll(sentence.Content, "-", " ")
	sentence.Content = strings.TrimSpace(sentence.Content)
}

func removeStopWords(locale string, words []string) []string {
	if len(words) <= 4 {
		return words
	}

	stopWords := string(ReadFile(GetResDir("locales", "stopwords.txt", locale)))

	var wordsToRemove []string

	for _, stopWord := range strings.Split(stopWords, "\n") {
		for _, word := range words {

			if !strings.Contains(stopWord, word) {
				continue
			}

			wordsToRemove = append(wordsToRemove, word)
		}
	}

	return Difference(words, wordsToRemove)
}

func (sentence Sentence) tokenize() (tokens []string) {
	tokens = strings.Fields(sentence.Content)

	for i, token := range tokens {
		tokens[i] = strings.ToLower(token)
	}

	tokens = removeStopWords(sentence.Locale, tokens)

	return
}

func (sentence Sentence) stem() (tokenizeWords []string) {
	locale := GetTagByName(sentence.Locale)

	if locale == "" {
		locale = "english"
	}

	tokens := sentence.tokenize()

	stemmer, err := snowball.New(locale)
	if err != nil {
		fmt.Println("Stemmer error", err)
		return
	}

	for _, tokenizeWord := range tokens {
		word := stemmer.Stem(tokenizeWord)
		tokenizeWords = append(tokenizeWords, word)
	}

	return
}

func (sentence Sentence) WordsBag(words []string) (bag []float64) {
	for _, word := range words {

		var valueToAppend float64
		if Contains(sentence.stem(), word) {
			valueToAppend = 1
		}

		bag = append(bag, valueToAppend)
	}

	return bag
}

func GetCoverage(writer http.ResponseWriter, _ *http.Request) {
	allowedHeaders := "Accept, Content-Type, Content-Length, Accept-Encoding, Authorization,Marboris-Token"
	writer.Header().Set("Access-Control-Allow-Origin", "*")
	writer.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
	writer.Header().Set("Access-Control-Allow-Headers", allowedHeaders)
	writer.Header().Set("Access-Control-Expose-Headers", "Authorization")

	defaultMessages, defaultIntents, defaultModules = GetMessages("en"), GetIntentsa("en"), GetModules("en")

	var coverage []LocaleCoverage

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

func getMessageCoverage(locale string) CoverageDetails {
	var notCoveredMessages []string

	for _, defaultMessage := range defaultMessages {
		message := GetMessageByTag(defaultMessage.Tag, locale)

		if message.Tag != defaultMessage.Tag {
			notCoveredMessages = append(notCoveredMessages, defaultMessage.Tag)
		}
	}

	coverage := calculateCoverage(len(notCoveredMessages), len(defaultMessages))

	return CoverageDetails{
		NotCovered: notCoveredMessages,
		Coverage:   coverage,
	}
}

func getIntentCoverage(locale string) CoverageDetails {
	var notCoveredIntents []string

	for _, defaultIntent := range defaultIntents {
		intent := GetIntentByTag(defaultIntent.Tag, locale)

		if intent.Tag != defaultIntent.Tag {
			notCoveredIntents = append(notCoveredIntents, defaultIntent.Tag)
		}
	}

	coverage := calculateCoverage(len(notCoveredIntents), len(defaultModules))

	return CoverageDetails{
		NotCovered: notCoveredIntents,
		Coverage:   coverage,
	}
}

func getModuleCoverage(locale string) CoverageDetails {
	var notCoveredModules []string

	for _, defaultModule := range defaultModules {
		module := GetModuleByTag(defaultModule.Tag, locale)

		if module.Tag != defaultModule.Tag {
			notCoveredModules = append(notCoveredModules, defaultModule.Tag)
		}
	}

	coverage := calculateCoverage(len(notCoveredModules), len(defaultModules))

	return CoverageDetails{
		NotCovered: notCoveredModules,
		Coverage:   coverage,
	}
}

func calculateCoverage(notCoveredLength, defaultLength int) int {
	return 100 * (defaultLength - notCoveredLength) / defaultLength
}

func GetNameByTag(tag string) string {
	for _, locale := range Locales {
		if locale.Tag != tag {
			continue
		}

		return locale.Name
	}

	return ""
}

func GetTagByName(name string) string {
	for _, locale := range Locales {
		if locale.Name != name {
			continue
		}

		return locale.Tag
	}

	return ""
}

func Exists(tag string) bool {
	for _, locale := range Locales {
		if locale.Tag == tag {
			return true
		}
	}

	return false
}

func GenerateToken() string {
	b := make([]byte, 30)
	crand.Read(b)

	fmt.Println("hey")
	return fmt.Sprintf("%x", b)
}

func HashToken(token string) []byte {
	bytes, _ := bcrypt.GenerateFromPassword([]byte(token), 14)
	return bytes
}

func ChecksToken(token string) bool {
	err := bcrypt.CompareHashAndPassword(authenticationHash, []byte(token))
	return err == nil
}

func AuthenticationFileExists() bool {
	_, err := os.Open(fileName)
	return err == nil
}

func SaveHash(hash string) {
	file, err := os.Create(fileName)
	if err != nil {
		panic(err)
	}

	defer file.Close()

	file.WriteString(hash)
}

func Authenticate() {
	if AuthenticationFileExists() {
		authenticationHash = ReadFile(fileName)
		return
	}

	token := GenerateToken()
	fmt.Printf("Your authentication token is: %s\n", color.FgLightGreen.Render(token))
	fmt.Println("Save it, you won't be able to get it again unless you generate a new one.")
	fmt.Println()

	hash := HashToken(token)
	SaveHash(string(hash))

	authenticationHash = hash
}

func WriteIntents(locale string, intents []Intent) {
	CacheIntents(locale, intents)

	bytes, _ := json.MarshalIndent(intents, "", "  ")

	file, err := os.Create(GetResDir("locales", "intents.json", locale))
	if err != nil {
		panic(err)
	}

	defer file.Close()

	file.Write(bytes)
}

func AddIntent(locale string, intent Intent) {
	intents := append(SerializeIntents(locale), intent)

	WriteIntents(locale, intents)

	fmt.Printf("Added %s intent.\n", color.FgMagenta.Render(intent.Tag))
}

func RemoveIntent(locale, tag string) {
	intents := SerializeIntents(locale)

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

func GetIntents(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")

	data := mux.Vars(r)

	json.NewEncoder(w).Encode(GetIntentsa(data["locale"]))
}

func CreateIntent(w http.ResponseWriter, r *http.Request) {
	allowedHeaders := "Accept, Content-Type, Content-Length, Accept-Encoding, Authorization,Marboris-Token"
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
	w.Header().Set("Access-Control-Allow-Headers", allowedHeaders)
	w.Header().Set("Access-Control-Expose-Headers", "Authorization")

	data := mux.Vars(r)

	token := r.Header.Get("Marboris-Token")
	if !ChecksToken(token) {
		json.NewEncoder(w).Encode(Error{
			Message: GetMessageu(data["locale"], "no permission"),
		})
		return
	}

	var intent Intent
	json.NewDecoder(r.Body).Decode(&intent)

	if intent.Responses == nil || intent.Patterns == nil {
		json.NewEncoder(w).Encode(Error{
			Message: GetMessageu(data["locale"], "patterns same"),
		})
		return
	}

	for _, _intent := range GetIntentsa(data["locale"]) {
		if _intent.Tag == intent.Tag {
			json.NewEncoder(w).Encode(Error{
				Message: GetMessageu(data["locale"], "tags same"),
			})
			return
		}
	}

	AddIntent(data["locale"], intent)

	json.NewEncoder(w).Encode(intent)
}

func DeleteIntent(w http.ResponseWriter, r *http.Request) {
	allowedHeaders := "Accept, Content-Type, Content-Length, Accept-Encoding, Authorization,Marboris-Token"
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "POST, GET, OPTIONS, PUT, DELETE")
	w.Header().Set("Access-Control-Allow-Headers", allowedHeaders)
	w.Header().Set("Access-Control-Expose-Headers", "Authorization")

	data := mux.Vars(r)

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

func GetLayers(locale string) Layers {
	return Layers{
		InputNodes: Rows(neuralNetworks[locale].Layers[0]),

		HiddenLayers: len(neuralNetworks[locale].Layers) - 2,

		OutputNodes: Columns(neuralNetworks[locale].Output),
	}
}

func GetTraining(locale string) Training {
	return Training{
		Rate:   neuralNetworks[locale].Rate,
		Errors: neuralNetworks[locale].Errors,
		Time:   neuralNetworks[locale].Time,
	}
}

func Serve(_neuralNetworks map[string]Network, port string) {
	neuralNetworks = _neuralNetworks // require

	router := mux.NewRouter()

	router.HandleFunc("/websocket", SocketHandle)

	router.HandleFunc("/api/{locale}/dashboard", GetDashboardData).Methods("GET")
	router.HandleFunc("/api/{locale}/intent", CreateIntent).Methods("POST")
	router.HandleFunc("/api/{locale}/intent", DeleteIntent).Methods("DELETE", "OPTIONS")
	router.HandleFunc("/api/{locale}/intents", GetIntents).Methods("GET")
	router.HandleFunc("/api/coverage", GetCoverage).Methods("GET")

	magenta := color.FgMagenta.Render
	fmt.Printf("\nServer listening on the port %s...\n", magenta(port))

	err := http.ListenAndServe(":"+port, router)
	if err != nil {
		panic(err)
	}
}

func SocketHandle(w http.ResponseWriter, r *http.Request) {
	conn, _ := upgrader.Upgrade(w, r, nil)
	fmt.Println(color.FgGreen.Render("A new connection has been opened"))

	for {

		msgType, msg, err := conn.ReadMessage()
		if err != nil {
			break
		}

		var request RequestMessage
		if err = json.Unmarshal(msg, &request); err != nil {
			continue
		}

		if reflect.DeepEqual(GetUserInformation(request.Token), Information{}) {
			SetUserInformation(request.Token, request.Information)
		}

		if request.Type == 0 {
			ExecuteModules(request.Token, request.Locale)

			message := GetMessage()
			if message != "" {

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

		response := Reply(request)
		if err = conn.WriteMessage(msgType, response); err != nil {
			continue
		}
	}
}

func Reply(request RequestMessage) []byte {
	var responseSentence, responseTag string

	if len(request.Content) > 500 {
		responseTag = "too long"
		responseSentence = GetMessageu(request.Locale, responseTag)
	} else {

		locale := request.Locale
		if !Exists(locale) {
			locale = "en"
		}

		responseTag, responseSentence = NewSentence(
			locale, request.Content,
		).Calculate(*cache, neuralNetworks[locale], request.Token)
	}

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

func ArticleCountries(name string) string {
	if name == "United States" {
		return "the " + name
	}

	return name
}

func SerializeCountries() (countries []Country) {
	err := json.Unmarshal(ReadFile(GetResDir("datasets", "countries.json")), &countries)
	if err != nil {
		fmt.Println(err)
	}

	return countries
}

func FindCountry(locale, sentence string) Country {
	for _, country := range countries {
		name, exists := country.Name[locale]

		if !exists {
			continue
		}

		if !strings.Contains(strings.ToLower(sentence), strings.ToLower(name)) {
			continue
		}

		return country
	}

	return Country{}
}

func SerializeMovies() (movies []Movie) {
	path := GetResDir("datasets", "movies.csv")
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

		rating, _ := strconv.ParseFloat(line[3], 64)

		movies = append(movies, Movie{
			Name:   line[1],
			Genres: strings.Split(line[2], "|"),
			Rating: rating,
		})
	}

	return
}

func SearchMovie(genre, userToken string) (output Movie) {
	for _, movie := range movies {
		userMovieBlacklist := GetUserInformation(userToken).MovieBlacklist

		if !Contains(movie.Genres, genre) || Contains(userMovieBlacklist, movie.Name) {
			continue
		}

		if reflect.DeepEqual(output, Movie{}) || movie.Rating > output.Rating {
			output = movie
		}
	}

	ChangeUserInformation(userToken, func(information Information) Information {
		information.MovieBlacklist = append(information.MovieBlacklist, output.Name)
		return information
	})

	return
}

func FindMoviesGenres(locale, content string) (output []string) {
	for i, genre := range MoviesGenres[locale] {
		if LevenshteinContains(strings.ToUpper(content), strings.ToUpper(genre), 2) {
			output = append(output, MoviesGenres["en"][i])
		}
	}

	return
}

func GetMessage() string {
	return message
}

func ExecuteModules(token, locale string) {
	fmt.Println(color.FgGreen.Render("Executing start modules.."))

	for _, module := range moduless {
		module.Action(token, locale)
	}
}

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

func AreaReplacer(locale, entry, response, _ string) (string, string) {
	country := FindCountry(locale, entry)

	if country.Currency == "" {
		responseTag := "no country"
		return responseTag, GetMessageu(locale, responseTag)
	}

	return AreaTag, fmt.Sprintf(response, ArticleCountriesm[locale](country.Name[locale]), country.Area)
}

func CapitalReplacer(locale, entry, response, _ string) (string, string) {
	country := FindCountry(locale, entry)

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

func CurrencyReplacer(locale, entry, response, _ string) (string, string) {
	country := FindCountry(locale, entry)

	if country.Currency == "" {
		responseTag := "no country"
		return responseTag, GetMessageu(locale, responseTag)
	}

	return CurrencyTag, fmt.Sprintf(response, ArticleCountriesm[locale](country.Name[locale]), country.Currency)
}

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

func MathReplacer(locale, entry, response, _ string) (string, string) {
	operation := FindMathOperation(entry)

	if operation == "" {
		responseTag := "don't understand"
		return responseTag, GetMessageu(locale, responseTag)
	}

	res, err := mathcat.Eval(operation)
	if err != nil {
		responseTag := "math not valid"
		return responseTag, GetMessageu(locale, responseTag)
	}

	decimals := FindNumberOfDecimals(locale, entry)
	if decimals == 0 {
		decimals = 6
	}

	result := res.FloatString(decimals)

	trailingZerosRegex := regexp.MustCompile(`\.?0+$`)
	result = trailingZerosRegex.ReplaceAllString(result, "")

	return MathTag, fmt.Sprintf(response, result)
}

func RegisterModules(locale string, _modules []Modulem) {
	modulesm[locale] = append(modulesm[locale], _modules...)
}

func GetModules(locale string) []Modulem {
	return modulesm[locale]
}

func GetModuleByTag(tag, locale string) Modulem {
	for _, module := range modulesm[locale] {
		if tag != module.Tag {
			continue
		}

		return module
	}

	return Modulem{}
}

func ReplaceContent(locale, tag, entry, response, token string) (string, string) {
	for _, module := range modulesm[locale] {
		if module.Tag != tag {
			continue
		}

		return module.Replacer(locale, entry, response, token)
	}

	return tag, response
}

func GenresReplacer(locale, entry, response, token string) (string, string) {
	genres := FindMoviesGenres(locale, entry)

	if len(genres) == 0 {
		responseTag := "no genres"
		return responseTag, GetMessageu(locale, responseTag)
	}

	ChangeUserInformation(token, func(information Information) Information {
		for _, genre := range genres {

			if Contains(information.MovieGenres, genre) {
				continue
			}

			information.MovieGenres = append(information.MovieGenres, genre)
		}
		return information
	})

	return GenresTag, response
}

func MovieSearchReplacer(locale, entry, response, token string) (string, string) {
	genres := FindMoviesGenres(locale, entry)

	if len(genres) == 0 {
		responseTag := "no genres"
		return responseTag, GetMessageu(locale, responseTag)
	}

	movie := SearchMovie(genres[0], token)

	return MoviesTag, fmt.Sprintf(response, movie.Name, movie.Rating)
}

func MovieSearchFromInformationReplacer(locale, _, response, token string) (string, string) {
	genres := GetUserInformation(token).MovieGenres
	if len(genres) == 0 {
		responseTag := "no genres saved"
		return responseTag, GetMessageu(locale, responseTag)
	}

	movie := SearchMovie(genres[rand.Intn(len(genres))], token)
	genresJoined := strings.Join(genres, ", ")
	return MoviesDataTag, fmt.Sprintf(response, genresJoined, movie.Name, movie.Rating)
}

func NameGetterReplacer(locale, _, response, token string) (string, string) {
	name := GetUserInformation(token).Name

	if strings.TrimSpace(name) == "" {
		responseTag := "don't know name"
		return responseTag, GetMessageu(locale, responseTag)
	}

	return NameGetterTag, fmt.Sprintf(response, name)
}

func NameSetterReplacer(locale, entry, response, token string) (string, string) {
	name := FindName(entry)

	if name == "" {
		responseTag := "no name"
		return responseTag, GetMessageu(locale, responseTag)
	}

	name = strings.Title(name)

	ChangeUserInformation(token, func(information Information) Information {
		information.Name = name
		return information
	})

	return NameSetterTag, fmt.Sprintf(response, name)
}

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

func LevenshteinDistance(first, second string) int {
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

func LevenshteinContains(sentence, matching string, rate int) bool {
	words := strings.Split(sentence, " ")
	for _, word := range words {
		if LevenshteinDistance(word, matching) <= rate {
			return true
		}
	}

	return false
}

func FindMathOperation(entry string) string {
	mathRegex := regexp.MustCompile(
		`((\()?(((\d+|pi)(\^\d+|!|.)?)|sqrt|cos|sin|tan|acos|asin|atan|log|ln|abs)( )?[+*\/\-x]?( )?(\))?[+*\/\-]?)+`,
	)

	operation := mathRegex.FindString(entry)

	operation = strings.Replace(operation, "x", "*", -1)
	return strings.TrimSpace(operation)
}

func FindNumberOfDecimals(locale, entry string) int {
	decimalsRegex := regexp.MustCompile(
		MathDecimals[locale],
	)
	numberRegex := regexp.MustCompile(`\d+`)

	decimals := numberRegex.FindString(decimalsRegex.FindString(entry))
	decimalsInt, _ := strconv.Atoi(decimals)

	return decimalsInt
}

func SerializeNames() (names []string) {
	namesFile := string(ReadFile(GetResDir("datasets", "names.txt")))

	names = append(names, strings.Split(strings.TrimSuffix(namesFile, "\n"), "\n")...)
	return
}

func FindName(sentence string) string {
	for _, name := range names {
		if !strings.Contains(strings.ToLower(" "+sentence+" "), " "+name+" ") {
			continue
		}

		return name
	}

	return ""
}

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
