package core

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

type Locale struct {
	Tag  string
	Name string
}

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

type Country struct {
	Name     map[string]string `json:"name"`
	Capital  string            `json:"capital"`
	Code     string            `json:"code"`
	Area     float64           `json:"area"`
	Currency string            `json:"currency"`
}

type Movie struct {
	Name   string
	Genres []string
	Rating float64
}

type Modules struct {
	Action func(string, string)
}

type Joke struct {
	ID        int64  `json:"id"`
	Type      string `json:"type"`
	Setup     string `json:"setup"`
	Punchline string `json:"punchline"`
}

type Modulem struct {
	Tag       string
	Patterns  []string
	Responses []string
	Replacer  func(string, string, string, string) (string, string)
	Context   string
}

type PatternTranslations struct {
	DateRegex string
	TimeRegex string
}
