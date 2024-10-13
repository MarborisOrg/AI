package core

import (
	"net/http"
	"time"

	"github.com/gorilla/websocket"

	gocache "github.com/patrickmn/go-cache"
)

var (
	defaultModules  []Modulem
	defaultIntents  []Intent
	defaultMessages []Message

	daysOfWeek = map[string]time.Weekday{
		"monday":    time.Monday,
		"tuesday":   time.Tuesday,
		"wednesday": time.Wednesday,
		"thursday":  time.Thursday,
		"friday":    time.Friday,
		"saturday":  time.Saturday,
		"sunday":    time.Sunday,
	}

	intents = map[string][]Intent{}

	userCache = gocache.New(5*time.Minute, 5*time.Minute)

	userInformation = map[string]Information{}

	Locales = []Locale{
		{
			Tag:  "en",
			Name: "english",
		},
	}

	fileName = getTempDir("Marboris-Authentication.txt")

	authenticationHash []byte

	upgrader = websocket.Upgrader{
		CheckOrigin: func(r *http.Request) bool {
			return true
		},
	}

	neuralNetworks map[string]Network

	cache = gocache.New(5*time.Minute, 5*time.Minute)

	countries = SerializeCountries()

	MoviesGenres = map[string][]string{
		"en": {
			"Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
			"Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
		},
	}
	movies = SerializeMovies()

	moduless []Modules
	message  string

	AdvicesTag = "advices"
	AreaTag    = "area"

	CapitalTag = "capital"

	ArticleCountriesm = map[string]func(string) string{}

	CurrencyTag = "currency"
	JokesTag    = "jokes"
	MathTag     = "math"

	modulesm = map[string][]Modulem{}

	GenresTag = "movies genres"

	MoviesTag = "movies search"

	MoviesAlreadyTag = "already seen movie"

	MoviesDataTag = "movies search from data"

	NameGetterTag = "name getter"

	NameSetterTag = "name setter"

	RandomTag          = "random number"
	PatternTranslation = map[string]PatternTranslations{
		"en": {
			DateRegex: `(of )?(the )?((after )?tomorrow|((today|tonight)|(next )?(monday|tuesday|wednesday|thursday|friday|saturday|sunday))|(\d{2}|\d)(th|rd|st|nd)? (of )?(january|february|march|april|may|june|july|august|september|october|november|december)|((\d{2}|\d)/(\d{2}|\d)))`,
			TimeRegex: `(at )?(\d{2}|\d)(:\d{2}|\d)?( )?(pm|am|p\.m|a\.m)`,
		},
	}

	rules []Rule

	RuleTranslations = map[string]RuleTranslation{
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

	MathDecimals = map[string]string{
		"en": `(\d+( |-)decimal(s)?)|(number (of )?decimal(s)? (is )?\d+)`,
	}
	names = SerializeNames()

	messages = map[string][]Message{}
	decimal  = "\\b\\d+([\\.,]\\d+)?"
)

// -----------------------------------------------------------

const (
	DontUnderstand = "don't understand"
	day            = time.Hour * 24
	jokeURL        = "https://official-joke-api.appspot.com/random_joke"
	adviceURL      = "https://api.adviceslip.com/advice"
)
