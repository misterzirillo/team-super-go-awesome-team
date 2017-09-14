# Project 1
Things about stuff here.

## Design document
[Google docs](https://docs.google.com/document/d/1AYEbssfIVpwn4aR2wP3jLsV7RXUbKTZNf2FqlRP2Dgc/edit?usp=sharing)

## Classifiers

_Why are linear models bad? Why are they good?_

### Ecoli dataset
- k-class problem
- 8 features
- continuous real 0..1
- 336 instances

#### Naive bayes
**prediction good** 

#### Decision tree
**prediction** _explore more_, maybe slow because many continuos features

#### K-nearest neighbor
**prediction good** for for real value features, research smoothing factor. May have an issue with some classifiers having too few instances in the set.

#### Logistic Regression
**prediction** expect similar results to naive bayes. _examine logReg handling k-class problem accuracy_.

#### Support vector machines
???

### Ionosphere dataset
- binary classification
- attrs 34
- values -1 to 1, continuous real
- 351 instances
- _data is clustered??_ we think so

#### Naive bayes
**prediction good** lots of features, no missing values

#### Decision trees
**prediction meh** many attributes

#### K-nearest neighbor
**prediction good** good sample distribution, seems fairly clustered

#### Logistic regression
**prediction good** binary class problem, clustered data

#### SVM
_todo_

### Fertility
- 10 attributes (with class)
- feature types are mixed
- 100 instances **low**

#### Naive bayes
**prediction poor** dataset is small, _research maybe pro that some features are discrete_

#### Decision trees
**prediction good** some discrete features

#### KNN
**prediction good** discrete features = clustering

#### Logistic regression
**prediction good** binary classifier

#### SVM
_todo_

### Magic dataset
- binary classification
- 10 attributes + class
- 
