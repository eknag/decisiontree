package decisiontree

type intSample struct {
	target   string
	features *map[string]int
}

func NewIntSample(target string, features map[string]int) *intSample {
	return &intSample{
		target:   target,
		features: &features,
	}
}

func (s *intSample) Feature(key string) (res int, found bool) {
	res, found = (*s.features)[key]
	return
}
