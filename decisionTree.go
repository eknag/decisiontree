package decisiontree

import "fmt"

type decisionTree struct {
	maxDepth      int
	currentDepth  int
	minSamples    int
	minGain       float64
	dataset       *intDataset
	split_feature string
	children      map[int]*decisionTree
	majorityClass int
	leaf          bool
}

func NewDecisionTree(maxDepth int,
	currentDepth int,
	minSamples int,
	minGain float64,
	dataset *intDataset) *decisionTree {

	majorityClass := dataset.MajorityClass()
	children := make(map[int]*decisionTree)
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

	return &decisionTree{
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

func (dt *decisionTree) Predict(sample *intSample) int {
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

func (dt *decisionTree) Depth() int {
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
