package core

import (
	"net/http"
	"time"

	"github.com/gorilla/websocket"

	gocache "github.com/patrickmn/go-cache"
)

// -----------------------------------------------------------
type Message struct {
	Tag      string   `json:"tag"`
	Messages []string `json:"messages"`
}
type Information struct {
	Name           string   `json:"name"`
	MovieGenres    []string `json:"movie_genres"`
	MovieBlacklist []string `json:"movie_blacklist"`
}
type Derivative struct {
	Delta      Matrix
	Adjustment Matrix
}
type (
	Matrix  [][]float64
	Network struct {
		Layers  []Matrix
		Weights []Matrix
		Biases  []Matrix
		Output  Matrix
		Rate    float64
		Errors  []float64
		Time    float64
		Locale  string
	}
)

type Sentence struct {
	Locale  string
	Content string
}

type Result struct {
	Tag   string
	Value float64
}

var (
	defaultModules  []Modulem
	defaultIntents  []Intent
	defaultMessages []Message
)

type LocaleCoverage struct {
	Tag      string   `json:"locale_tag"`
	Language string   `json:"language"`
	Coverage Coverage `json:"coverage"`
}

type Coverage struct {
	Modules  CoverageDetails `json:"modules"`
	Intents  CoverageDetails `json:"intents"`
	Messages CoverageDetails `json:"messages"`
}

type CoverageDetails struct {
	NotCovered []string `json:"not_covered"`
	Coverage   int      `json:"coverage"`
}

var intents = map[string][]Intent{}

type Intent struct {
	Tag       string   `json:"tag"`
	Patterns  []string `json:"patterns"`
	Responses []string `json:"responses"`
	Context   string   `json:"context"`
}

type Document struct {
	Sentence Sentence
	Tag      string
}

var userCache = gocache.New(5*time.Minute, 5*time.Minute)

const DontUnderstand = "don't understand"

var userInformation = map[string]Information{}

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

var fileName = getTempDir("Marboris-Authentication.txt")

var authenticationHash []byte

type Error struct {
	Message string `json:"message"`
}

type DeleteRequest struct {
	Tag string `json:"tag"`
}

type Dashboard struct {
	Layers   Layers   `json:"layers"`
	Training Training `json:"training"`
}

type Layers struct {
	InputNodes   int `json:"input"`
	HiddenLayers int `json:"hidden"`
	OutputNodes  int `json:"output"`
}

type Training struct {
	Rate   float64   `json:"rate"`
	Errors []float64 `json:"errors"`
	Time   float64   `json:"time"`
}

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		return true
	},
}

type RequestMessage struct {
	Type        int         `json:"type"` // 0 for handshakes and 1 for messages
	Content     string      `json:"content"`
	Token       string      `json:"user_token"`
	Locale      string      `json:"locale"`
	Information Information `json:"information"`
}

type ResponseMessage struct {
	Content     string      `json:"content"`
	Tag         string      `json:"tag"`
	Information Information `json:"information"`
}

var (
	neuralNetworks map[string]Network

	cache = gocache.New(5*time.Minute, 5*time.Minute)
)

type Country struct {
	Name     map[string]string `json:"name"`
	Capital  string            `json:"capital"`
	Code     string            `json:"code"`
	Area     float64           `json:"area"`
	Currency string            `json:"currency"`
}

var countries = SerializeCountries()

type Movie struct {
	Name   string
	Genres []string
	Rating float64
}

var (
	MoviesGenres = map[string][]string{
		"en": {
			"Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
			"Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
		},
	}
	movies = SerializeMovies()
)

type Modules struct {
	Action func(string, string)
}

var (
	moduless []Modules
	message  string
)

const adviceURL = "https://api.adviceslip.com/advice"

var (
	AdvicesTag = "advices"
	AreaTag    = "area"
)

var (
	CapitalTag = "capital"

	ArticleCountriesm = map[string]func(string) string{}
)

var CurrencyTag = "currency"

const jokeURL = "https://official-joke-api.appspot.com/random_joke"

var JokesTag = "jokes"

type Joke struct {
	ID        int64  `json:"id"`
	Type      string `json:"type"`
	Setup     string `json:"setup"`
	Punchline string `json:"punchline"`
}

var MathTag = "math"

type Modulem struct {
	Tag       string
	Patterns  []string
	Responses []string
	Replacer  func(string, string, string, string) (string, string)
	Context   string
}

var modulesm = map[string][]Modulem{}
var (
	GenresTag = "movies genres"

	MoviesTag = "movies search"

	MoviesAlreadyTag = "already seen movie"

	MoviesDataTag = "movies search from data"
)

var (
	NameGetterTag = "name getter"

	NameSetterTag = "name setter"
)

var (
	RandomTag          = "random number"
	PatternTranslation = map[string]PatternTranslations{
		"en": {
			DateRegex: `(of )?(the )?((after )?tomorrow|((today|tonight)|(next )?(monday|tuesday|wednesday|thursday|friday|saturday|sunday))|(\d{2}|\d)(th|rd|st|nd)? (of )?(january|february|march|april|may|june|july|august|september|october|november|december)|((\d{2}|\d)/(\d{2}|\d)))`,
			TimeRegex: `(at )?(\d{2}|\d)(:\d{2}|\d)?( )?(pm|am|p\.m|a\.m)`,
		},
	}
)

type PatternTranslations struct {
	DateRegex string
	TimeRegex string
}

type Rule func(string, string) time.Time

var rules []Rule

const day = time.Hour * 24

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

var MathDecimals = map[string]string{
	"en": `(\d+( |-)decimal(s)?)|(number (of )?decimal(s)? (is )?\d+)`,
}

var names = SerializeNames()

// -----------------------------------------------------------
var messages = map[string][]Message{}
var decimal = "\\b\\d+([\\.,]\\d+)?"
