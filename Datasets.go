package main

import (
	"errors"
	"math"
	"strconv"
)

// type CategoricalFeature interface {
// 	Filter(indexes []int) (CategoricalFeature, error)
// 	Entropy() float32
// 	NumSamples() (int()
// 	Get(index int) interface{}
// 	Unique() map[intfunc main() {
// 	Append(item *interface {})
// }

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

type intFeature []int

func (f *intFeature) Filter(mask []bool) (res *intFeature, err error) {

	if len(mask) != len(*f) {
		err = errors.New("indexes must mach the feature length in filter")
		return
	}
	array := make(intFeature, 0)

	for i, v := range mask {
		if v {
			array = append(array, (*f)[i])
		}
	}
	res = &array
	return
}

func (f *intFeature) NumSamples() (res int) {
	if f == nil {
		return
	}

	res = len(*f)
	return
}

func (f *intFeature) Get(idx int) (res int) {
	res = (*f)[idx]
	return
}

func (f *intFeature) Unique() (res map[int]int) {
	if f == nil {
		return
	}
	res = make(map[int]int)
	for _, v := range *f {
		res[v]++
	}
	return
}

func (f *intFeature) Entropy() (res float64) {
	elements := f.Unique()
	if elements == nil {
		return
	}
	for _, count := range elements {
		p := float64(count) / float64(f.NumSamples())
		res -= p * math.Log2(p)
	}
	return
}

func (d *intFeature) String() (res string) {
	if d == nil {
		return
	}
	res += "["
	for i, v := range *d {
		if i != 0 {
			res += ", "
		}
		res += strconv.Itoa(v)
	}
	res += "]"
	return
}

type intDataset struct {
	features    *map[string]*intFeature
	numFeatures int
	numSamples  int
	target      string
}

func NewIntDataset(features map[string]*intFeature, target string) (res *intDataset) {
	res = &intDataset{
		features:    &features,
		numFeatures: len(features),
		numSamples:  len(*features[target]),
		target:      target,
	}
	for _, feature := range features {
		if res.NumSamples() != feature.NumSamples() {
			panic("num samples not equal while initializing dataset")
		}
	}
	return
}

func (d intDataset) NumFeatures() int {
	return d.numFeatures
}

func (d intDataset) NumSamples() int {
	return d.numSamples
}

func (d intDataset) GetFeature(key string) (res *intFeature, found bool) {
	res, found = (*d.features)[key]
	return
}

func (d intDataset) SubsetByFeature(feature_name string) (res map[int]*intDataset) {
	feature, found := d.GetFeature(feature_name)
	samples_seen := 0
	if !found {
		return
	}

	res = make(map[int]*intDataset)
	for v := range feature.Unique() {
		mask := make([]bool, d.NumSamples())
		num_samples := 0
		for i, _ := range mask {
			mask[i] = v == (*feature)[i]
			if mask[i] {
				num_samples++
			}
		}
		//fmt.Println("num samples :" + strconv.Itoa(num_samples))

		new_features := make(map[string]*intFeature)
		for key, feature := range *d.features {
			sub_feature, _ := feature.Filter(mask)
			new_features[key] = sub_feature
		}
		subset := intDataset{
			features:    &new_features,
			numFeatures: d.numFeatures,
			numSamples:  num_samples,
			target:      d.target,
		}
		//fmt.Println("subset =", subset.String())
		res[v] = &subset
		samples_seen += num_samples
	}
	if samples_seen != d.NumSamples() {
		panic("samples_seen != d.NumSamples()")
	}
	return
}

func (d intDataset) MutualInformation(f string) (res float64, found bool) {
	feature, found1 := d.GetFeature(f)
	target, found2 := d.GetFeature(d.target)
	found = found1 && found2
	if !found {
		return
	}
	num_samples := d.NumSamples()
	for v1 := range feature.Unique() {
		mask := make([]bool, d.NumSamples())
		num_subsamples := 0
		for i, _ := range mask {
			mask[i] = v1 == (*feature)[i]
			if mask[i] {
				num_subsamples++
			}
		}
		subset, _ := target.Filter(mask)
		res -= float64(num_subsamples) / float64(num_samples) * subset.Entropy()
	}
	res += target.Entropy()
	return

}

func (d intDataset) MajorityClass() (res int) {
	t, _ := d.GetFeature(d.target)
	elements := t.Unique()
	max := 0
	var empty bool
	for v, count := range elements {
		if count > max {
			max = count
			res = v
		}
	}
	if empty {
		panic("majority class called on empty dataset")
	}
	return
}

func (d intDataset) String() (res string) {
	res += "target: " + d.target + "\n"
	res += "numFeatures: " + strconv.Itoa(d.numFeatures) + "\n"
	res += "numSamples: " + strconv.Itoa(d.numSamples) + "\n"
	for key, feature := range *d.features {
		res += key + ": " + feature.String() + "\n"
	}
	return
}

// func main() {
// 	target := intFeature{1, 1, 0, 1}
// 	f1 := intFeature{1, 1, 2, 2}
// 	f2 := intFeature{3, 3, 3, 4}
// 	d := intDataset{
// 		features:    &map[string]*intFeature{"target": &target, "f1": &f1, "f2": &f2},
// 		numFeatures: 2,
// 		numSamples:  4,
// 		target:      "target",
// 	}
// 	fmt.Println(d.MutualInformation("target"))
// 	fmt.Println(d.MutualInformation("f1"))
// 	fmt.Println(d.MutualInformation("f2"))

// 	fmt.Println(d.SubsetByFeature("f1"))
// 	fmt.Println(f1)
// }

// func (f *intFeature) Append()
// type Dataset interface {
// 	GetFeature(key string) (string, error)
// 	SetFeature(key, value intFeature) error
// 	DeleteFeature(key string) error
// 	Features() []string
// 	NumSamples() int
// 	NumFeatures() int
// 	FilterSamples(indexes []int) (Dataset, error)
// }

//operations we need to do on our dataset
//---- Initializing ---
//1. Initialize
//2. Add a sample
//3. Add a feature
//-- best split selection
//4. Get a feature
//5. Calculate the entropy of a feature
//6. Calculate the mutual information of two features
//-- subsetting
//7. Partition based on a feature
//8. subset based on an array of indexes
//-- error checking
//9. ensure all features are the same length
//10. ensure all samples are the same length
