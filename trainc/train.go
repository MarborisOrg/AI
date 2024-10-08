package trainc

import (
	"encoding/json"

	"github.com/tebeka/snowball"
	"math/rand"
	"os"
	"regexp"
	"sort"
	"strings"

	"fmt"
	"github.com/gookit/color"

	"errors"
	"github.com/soudy/mathcat"
	"gopkg.in/cheggaaa/pb.v1"
	"io"
	"math"
	"net/http"
	"strconv"
	"time"

	"bufio"
	"encoding/csv"
	"log"
	"reflect"
)

// GetUserInformation returns the information of a user with his token
func GetUserInformation(token string) Information {
	return userInformation[token]
}

// RegisterModules registers an array of modulesm into the map
func RegisterModules(locale string, _modules []Modulem) {
	modulesm[locale] = append(modulesm[locale], _modules...)
}

// AreaTag is the intent tag for its module
var AreaTag = "area"

// Country is the serializer of the countries.json file in the res folder
type Country struct {
	Name     map[string]string `json:"name"`
	Capital  string            `json:"capital"`
	Code     string            `json:"code"`
	Area     float64           `json:"area"`
	Currency string            `json:"currency"`
}

// SerializeCountries returns a list of countries, serialized from `res/datasets/countries.json`
func SerializeCountries() (countries []Country) {
	err := json.Unmarshal(ReadFile("res/datasets/countries.json"), &countries)
	if err != nil {
		fmt.Println(err)
	}

	return countries
}

var countries = SerializeCountries()

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

// MathTag is the intent tag for its module
var MathTag = "math"

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

type Message struct {
	Tag      string   `json:"tag"`
	Messages []string `json:"messages"`
}

var messages = map[string][]Message{}

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
		rand.NewSource(time.Now().UnixNano()) // Seed
		return message.Messages[rand.Intn(len(message.Messages))]
	}

	return ""
}

// package modules

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

// Movie is the serializer from res/datasets/movies.csv
type Movie struct {
	Name   string
	Genres []string
	Rating float64
}

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

// FindMoviesGenres returns an array of genres found in the entry string
func FindMoviesGenres(locale, content string) (output []string) {
	for i, genre := range MoviesGenres[locale] {
		if LevenshteinContains(strings.ToUpper(content), strings.ToUpper(genre), 2) {
			output = append(output, MoviesGenres["en"][i])
		}
	}

	return
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

// A Rule is a function that takes the given sentence and tries to parse a specific
// rule to return a date, if not, the date is empty.
type Rule func(string, string) time.Time

var rules []Rule

// RegisterRule takes a rule in parameter and register it to the array of rules
func RegisterRule(rule Rule) {
	rules = append(rules, rule)
}

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

func init() {
	// Register the rules
	RegisterRule(RuleToday)
	RegisterRule(RuleTomorrow)
	RegisterRule(RuleDayOfWeek)
	RegisterRule(RuleNaturalDate)
	RegisterRule(RuleDate)
}

func Contains(slice []string, text string) bool {
	for _, item := range slice {
		if item == text {
			return true
		}
	}

	return false
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

func CacheIntents(locale string, _intents []Intent) {
	intents[locale] = _intents
}

func SerializeIntents(locale string) (_intents []Intent) {
	err := json.Unmarshal(ReadFile("res/locales/"+locale+"/intents.json"), &_intents)
	if err != nil {
		panic(err)
	}

	CacheIntents(locale, _intents)

	return _intents
}

type Modulem struct {
	Tag       string
	Patterns  []string
	Responses []string
	Replacer  func(string, string, string, string) (string, string)
	Context   string
}

var modulesm = map[string][]Modulem{}

func GetModules(locale string) []Modulem {
	return modulesm[locale]
}

var intents = map[string][]Intent{}

type Intent struct {
	Tag       string   `json:"tag"`
	Patterns  []string `json:"patterns"`
	Responses []string `json:"responses"`
	Context   string   `json:"context"`
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

type Sentence struct {
	Locale  string
	Content string
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

type Document struct {
	Sentence Sentence
	Tag      string
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

func removeStopWords(locale string, words []string) []string {

	if len(words) <= 4 {
		return words
	}

	stopWords := string(ReadFile("res/locales/" + locale + "/stopwords.txt"))

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

var Locales = []Locale{
	{
		Tag:  "en",
		Name: "english",
	},
}

type Locale struct {
	Tag  string
	Name string
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

func Index(slice []string, text string) int {
	for i, item := range slice {
		if item == text {
			return i
		}
	}

	return 0
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

func TrainData(locale string) (inputs, outputs [][]float64) {
	words, classes, documents := Organize(locale)

	for _, document := range documents {
		outputRow := make([]float64, len(classes))
		bag := document.Sentence.WordsBag(words)

		outputRow[Index(classes, document.Tag)] = 1

		inputs = append(inputs, bag)
		outputs = append(outputs, outputRow)
	}

	return inputs, outputs
}

type Matrix [][]float64

func CreateMatrix(rows, columns int) (matrix Matrix) {
	matrix = make(Matrix, rows)

	for i := 0; i < rows; i++ {
		matrix[i] = make([]float64, columns)
	}

	return
}

func Columns(matrix Matrix) int {
	return len(matrix[0])
}

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

func Rows(matrix Matrix) int {
	return len(matrix)
}

func CreateNetwork(locale string, rate float64, input, output Matrix, hiddensNodes ...int) Network {
	input = append([][]float64{
		make([]float64, len(input[0])),
	}, input...)
	output = append([][]float64{
		make([]float64, len(output[0])),
	}, output...)

	inputMatrix := input
	layers := []Matrix{inputMatrix}

	for _, hiddenNodes := range hiddensNodes {
		layers = append(layers, CreateMatrix(len(input), hiddenNodes))
	}

	layers = append(layers, output)

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

func ErrorNotSameSize(matrix, matrix2 Matrix) {
	if Rows(matrix) != Rows(matrix2) && Columns(matrix) != Columns(matrix2) {
		panic("These two matrices must have the same dimension.")
	}
}

func ApplyFunctionWithIndex(matrix Matrix, fn func(i, j int, x float64) float64) Matrix {
	for i := 0; i < Rows(matrix); i++ {
		for j := 0; j < Columns(matrix); j++ {
			matrix[i][j] = fn(i, j, matrix[i][j])
		}
	}

	return matrix
}

func Sum(matrix, matrix2 Matrix) (resultMatrix Matrix) {
	ErrorNotSameSize(matrix, matrix2)

	resultMatrix = CreateMatrix(Rows(matrix), Columns(matrix))

	return ApplyFunctionWithIndex(matrix, func(i, j int, x float64) float64 {
		return matrix[i][j] + matrix2[i][j]
	})
}

func ApplyFunction(matrix Matrix, fn func(x float64) float64) Matrix {
	return ApplyFunctionWithIndex(matrix, func(i, j int, x float64) float64 {
		return fn(x)
	})
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
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

type Derivative struct {
	Delta      Matrix
	Adjustment Matrix
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

	resultMatrix = CreateMatrix(Rows(matrix), Columns(matrix))

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

func MultipliesByTwo(x float64) float64 {
	return 2 * x
}

func SubtractsOne(x float64) float64 {
	return x - 1
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

func ApplyRate(matrix Matrix, rate float64) Matrix {
	return ApplyFunction(matrix, func(x float64) float64 {
		return rate * x
	})
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

func CreateNeuralNetwork(locale string) (neuralNetwork Network) {

	saveFile := "res/locales/" + locale + "/training.json"

	inputs, outputs := TrainData(locale)

	neuralNetwork = CreateNetwork(locale, 0.1, inputs, outputs, 50)
	neuralNetwork.Train(200)

	neuralNetwork.Save(saveFile)

	return
}
