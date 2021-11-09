package decisiontree

import (
	"errors"
	"math"
	"strconv"
)

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
