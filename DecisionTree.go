package main

import (
	"fmt"
)

type DecisionTree struct {
	maxDepth      int
	currentDepth  int
	minSamples    int
	minGain       float64
	dataset       *intDataset
	split_feature string
	children      map[int]*DecisionTree
	majorityClass int
	leaf          bool
}

func NewDecisionTree(maxDepth int,
	currentDepth int,
	minSamples int,
	minGain float64,
	dataset *intDataset) *DecisionTree {

	majorityClass := dataset.MajorityClass()
	children := make(map[int]*DecisionTree)
	leaf := false
	if currentDepth == maxDepth || dataset.NumSamples() < minSamples {
		leaf = true
	}

	best_split := ""
	best_gain := 0.0
	for f_name, _ := range *(dataset.features) {
		if f_name == dataset.target {
			continue
		} else if best_split == "" {
			best_split = f_name
			best_gain, _ = dataset.MutualInformation(f_name)
		} else {
			gain, _ := dataset.MutualInformation(f_name)
			if gain > best_gain {
				best_split = f_name
				best_gain = gain
			}
		}
	}
	if best_gain <= minGain {
		leaf = true
	}
	if !leaf {
		child_datasets := dataset.SubsetByFeature(best_split)
		for split_value, child_dataset := range child_datasets {
			child := NewDecisionTree(maxDepth,
				currentDepth+1,
				minSamples,
				minGain,
				child_dataset)
			children[split_value] = child
		}
	}

	return &DecisionTree{
		maxDepth:      maxDepth,
		currentDepth:  currentDepth,
		minSamples:    minSamples,
		minGain:       minGain,
		dataset:       dataset,
		split_feature: best_split,
		children:      children,
		majorityClass: majorityClass,
		leaf:          leaf,
	}
}

func (dt *DecisionTree) Predict(sample *intSample) int {
	if dt.leaf {
		return dt.majorityClass
	}
	split_feature := dt.split_feature
	split_value, _ := sample.Feature(split_feature)
	child, found := dt.children[split_value]
	if !found {
		fmt.Println("Warning: no child found for sample")
		return dt.majorityClass
	}
	return child.Predict(sample)
}

func (dt *DecisionTree) Depth() int {
	if dt.leaf {
		return 0
	}
	max_depth := 1
	for _, child := range dt.children {
		depth := child.Depth() + 1
		if depth > max_depth {
			max_depth = depth
		}
	}
	return max_depth
}

type DecisionTreeLearner struct {
	maxDepth   int
	minSamples int
	minGain    float64
	dataset    *intDataset
	root       *DecisionTree
}

func NewDecisionTreeLearner(maxDepth int,
	minSamples int,
	minGain float64,
	dataset *intDataset) *DecisionTreeLearner {
	tree := NewDecisionTree(maxDepth, 0, minSamples, minGain, dataset)
	return &DecisionTreeLearner{maxDepth: maxDepth,
		minSamples: minSamples,
		minGain:    minGain,
		dataset:    dataset,
		root:       tree}
}

func (dtl *DecisionTreeLearner) Predict(sample *intSample) int {
	return dtl.root.Predict(sample)
}

func (dtl *DecisionTreeLearner) Depth() int {
	return dtl.root.Depth()
}

func main() {
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
