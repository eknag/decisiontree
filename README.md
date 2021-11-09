# golang-decisiontree

## A simple decision tree algorithm that should definitely be used by no one

As a first exercise, while learning golang, I decided to implement a regular categorical decision tree. I'm sure the style is horrible and many golang conventions are violated. Additionally, I didn't teach myself how to package golang files appropriately, so everything is under package main.

## intSample.go
* intSample - this struct holds a single data point for use in prediction
    * NewIntSample - init function (not a method) 
    * Feature - method returns the value associated with the named feature

## intFeature.go
* intFeature - this is just an int slice
    * Filter - this method takes in an equal length []bool mask and returns the filtered feature
    * NumSamples - Returns the length of the feature
    * Get - gets the element in position idx
    * Unique - returns a dictionary mapping the unique elements to their counts
    * Entropy - calculates the entropy of the feature using log2
    * String - converts the feature to a string for printing
   
## intDataset.go 
* intDataset - This is the main dataset type used as an input to the decision tree
    * Note - I might want to consider breaking out the target from the features.
    * NewIntDataset - initialization function (not a method)
    * NumFeatures - returns the number of features (including the target) included in the dataset
    * NumSamples - the number of training samples provided
    * GetFeature - returns the intFeature corresponding to the named feature
    * SubsetByFeature - takes in a feature name, returns a partition of the dataset based on the feature
    * MutualInformation - calculates the mutual information or information gan between the target and the supplied feature
    * MajorityClass - returns the most commonly occurring class in the target
    * String - String method for printing


## decisionTree.go
* decisionTree - this is a helper type used to recursively form the tree, you can mostly ignore it
    * NewDecisionTree - init function (not method), called by NewDecisionTreeLearner
    * Predict - takes in an intSample and provides a prediction
    * Depth - returns the depth of the tree
  
## DecisionTreeLearner.go
* DecisionTreeLearner - the main type of the project
    * NewDecisionTreeLearner - takes in tree parameters and a dataset and initializes a new trained tree
    * Predict - predicts the class of an intSample
    * Depth - returns the depth of the decision tree
