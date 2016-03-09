# Machine-Learning
All Supervised Learning experiments; including Gaussian Naive Bayes classifiers, Perceptrons, SVM Models, and Multiclass Artificial Neural Network and Unsupervised Learning experiment using K-means algorithm and K-NN

# Clustering Using K-Means Algorithm and Euclidean Distance {k-means.py}

# Gaussian Naive Bayes Classifier {naive-bayes.py}
I have used the same spambase dataset used by the SVM classifier
For this task, I performed a supervised learning using Naive Bayes assuming a Gaussian distribution and a conditional independence of the attributes.
The code is pretty intuitive to follow, I have also added comment on every step.

# Artificial Neural Network     {nn.py}
This ANN has one input layer, one hidden layer and one output layer The ANN classifies the letter recognition dataset from: http://archive.ics.uci.edu/ml/datasets/Letter+Recognition Which has 20,000 instances with 16 features each.

The input layer has a 16 + 1 number of units including the bias input unit(+1) The hidden layer has a dynamic number of units, called n to make a total of (n+1) units including the bias The output layer consist of 26 outputs depicting each letter of the alphabet. (A-Z)

This ANN uses the back propagation algorithm for learning, with momentum and weight decay that can be added easily This serves as a homework submission for CS 545 - Machine Learning at Portland State University.

This code also plots a graph at the end of each run. Its very intuitive, once you understand how ANN works.

# Support Vector Machine Implementation using Python's SCIKIT-LEARN svm package {svm-exp.py}
I carried out a binary classification on the spam dataset from https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/

I have implemented the linear SVM, which is pretty straightforward. Also, I have carried out a 10-fold cross validation to determine the best C parameter (i.e the C with the highest average accuracy)

Also in this experiment, I have done some feature selection experiments (SVM-weighted feature selection and Random Feature Selection) to determine the best features used for the classification by the SVM.

# Perceptron {perceptron.py}
In this program, I have implemented a classification system using perceptrons. 
For this implementation, I used the same letter-recognition dataset used in the ANN implementation.

The perceptron's are trained using the stochastic gradient descent method and the predicition is done using an all pairs method.

Most implementations of perceptrons usually use the one vs all method for prediction.

For the task of letter recognition(A-Z), i have trained 325 perceptrons. (A vs B, A vs C, ..., A vs Z, B vs C, ...., C vs D , ....)
For prediction each perceptron votes the based on the instance passed, the letter with the most votes is eventually selected.

This code, is slow, and it is need of some corrections, I plan to post a much faster and efficient version soon enough.
It's is easy to understand.
