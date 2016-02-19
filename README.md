# Machine-Learning
All experiments, including Naive Bayes classifiers, SVM Models, and Multiclass Artificial Neural Network

# Artificial Neural Network     {nn.py}
This ANN has one input layer, one hidden layer and one output layer The ANN classifies the letter recognition dataset from: http://archive.ics.uci.edu/ml/datasets/Letter+Recognition Which has 20,000 instances with 16 features each.

The input layer has a 16 + 1 number of units including the bias input unit(+1) The hidden layer has a dynamic number of units, called n to make a total of (n+1) units including the bias The output layer consist of 26 outputs depicting each letter of the alphabet. (A-Z)

This ANN uses the back propagation algorithm for learning, with momentum and weight decay that can be added easily This serves as a homework submission for CS 545 - Machine Learning at Portland State University.

This code also plots a graph at the end of each run. Its very intuitive, once you understand how ANN works.

# Support Vector Machine Implementation using Python's SCIKIT-LEARN svm package {svm-exp.py}
I carried out a binary classification on the spam dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/

I have implemented the linear SVM, which is pretty straightforward. Also, I have carried out a 10-fold cross validation to determine the best C parameter (i.e the C with the highest average accuracy)

Also in this experiment, I have done some feature selection experiments (SVM-weighted feature selection and Random Feature Selection) to determine the best features used for the classification by the SVM.
