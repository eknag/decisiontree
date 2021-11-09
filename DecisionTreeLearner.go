package decisiontree

type DecisionTreeLearner struct {
	maxDepth   int
	minSamples int
	minGain    float64
	dataset    *intDataset
	root       *decisionTree
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
