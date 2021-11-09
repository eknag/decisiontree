package decisiontree

import (
	"fmt"
	"testing"
)

func TestMain(m *testing.M) {
	target := intFeature{1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 2}
	f1 := intFeature{1, 1, 2, 2, 3, 3, 3, 2, 3, 3, 1}
	f2 := intFeature{3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4}
	features := map[string]*intFeature{
		"target": &target,
		"f1":     &f1,
		"f2":     &f2,
	}
	d := NewIntDataset(features, "target")
	tree := NewDecisionTreeLearner(5, 0, 0.0, d)

	test_sample := NewIntSample("target", map[string]int{
		"f1": 1,
		"f2": 1,
	})
	fmt.Println(tree.Predict(test_sample))

}
