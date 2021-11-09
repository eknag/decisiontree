package decisiontree

import "strconv"

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
