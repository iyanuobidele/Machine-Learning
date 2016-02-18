# Homework 3: Classification using a Linear SVM
# Spam and No Spam classification.

# Import necessary libs
from sklearn import svm, metrics
import random
import matplotlib.pyplot as plt
import numpy as np
import warnings
import heapq
import csv


class SVM(object):
    def __init__(self):
        self.c = 0
        self.test = []
        self.train = []

    @staticmethod
    def partition(sequence, fold, position):
        # :sequence: NP array to be divided
        # :fold: No of divisions wanted
        # :position: Which particular division
        if len(sequence) % fold != 0:
            print "You cannot split this set evenly"
        else:
            # Simple algorithm to get the Start & End index position of the division
            start_index = ((position-1) * (len(sequence)/fold))
            end_index = start_index + (len(sequence)/fold)
            # Validation Set depending on the fold position
            validation_set = sequence[start_index:end_index]
            # Training set
            others = np.delete(sequence, np.s_[start_index:end_index], 0)
            return validation_set, others

    @staticmethod
    def match_values(array1, array2):
        """
        :param array1: First array
        :param array2: Second array
        :return: return number of matches
        """
        if len(array1) != len(array2):
            print "Values do not match !"
        else:
            counter = 0
            for i in range(len(array1)):
                if array1[i] == array2[i]:
                    counter += 1
            return counter                  # Return the number of matches

    @staticmethod
    def precision_recall_calc(score, target, threshold):
        """
        :param score: Decision values
        :param target: Expected
        :param threshold: Threshold to calculate correct values
        :return: precision and recall
        """
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0

        # TP, FP, FN, TN calculations
        for i in range(len(score)):
            if score[i] >= threshold and target[i] == 1:
                true_positives += 1
            if score[i] < threshold and target[i] == 0:
                true_negatives += 1
            if score[i] >= threshold and target[i] == 0:
                false_positives += 1
            if score[i] < threshold and target[i] == 1:
                false_negatives += 1

        # Simple Calculations of accuracy, recall, precision and false positive rate
        precision = float(true_positives) / (true_positives + false_positives)
        recall = float(true_positives) / (true_positives + false_negatives)
        fpr = float(false_positives) / (false_positives + true_negatives)
        accuracy = float(true_positives + true_negatives) / len(score)

        return accuracy, precision, recall, fpr

    def experiments(self, one=True, two=True, three=True):
        # 10-fold
        fold = 10

        # Training Data in 1810 by 58 including target
        with open(r'C:\Users\Iyanu\Desktop\Python\ML\data\3\s_train.data','r') as dest_f:
            data_iter = csv.reader(dest_f,
                                   delimiter=',',
                                   quotechar='"')
            data = [data for data in data_iter]
        self.train = np.asarray(data, dtype=float)

        # Test Data in 1810 by 58 including target
        with open(r'C:\Users\Iyanu\Desktop\Python\ML\data\3\s_test.data','r') as tdest_f:
            tdata_iter = csv.reader(tdest_f,
                                   delimiter=',',
                                   quotechar='"')
            tdata = [tdata for tdata in tdata_iter]
        self.test = np.asarray(tdata, dtype=float)

        # Shuffle training data
        np.random.shuffle(self.train)
        # np.random.shuffle(self.train)
        # np.random.shuffle(self.train)

        # Using 10-fold cross validation to find out the best c-parameter for the svm
        c_array = np.linspace(start=0.1, stop=1, num=20)    # Possible C values to test with. Evenly spaced 20 values
        temp_array = []
        avg_c_acc = []
        for i in range(len(c_array)):    # Outer loop for all Possible C values
            for ii in range(1, fold+1):  # Loop for Validation
                validation_set, training_set = self.partition(self.train, fold, ii)  # Divide training Set
                classifier = svm.SVC(C=c_array[i], kernel='linear', shrinking=True)                            # Initialise with first C class
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    classifier.fit(training_set[:, 0:57], training_set[:, 57:58])         # Fitting training set
                predicted = classifier.predict(validation_set[:, 0:57])             # Predict validation set
                matches = self.match_values(predicted, validation_set[:, 57:58])          # Calc Accuracy
                acc = float(matches)/len(validation_set)
                temp_array.append(acc)                                             # Accuracies for each validation set
            avg_acc = np.average(temp_array)                                       # Average of 10 accuracies for C of i
            temp_array = []                                                        # empty the array before next loop
            avg_c_acc.append(avg_acc)                                              # Store the average acc of C of i

        index = np.argmax(avg_c_acc)            # index of highest accuracy, the same as index of C value in C array
        self.c = c_array[index]                 # C*
        # print avg_c_acc
        # print c_array[index], max(avg_c_acc)    # Print C value w/ highest accuracy and Avg Accuracy of that C value

        # Now that we know the best C parameter, we can fit all training set, then do a prediction
        t_classifier = svm.SVC(C=self.c, kernel='linear', shrinking=True)          # Creating an object of the linear SVC using our C*
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        # This returns the decision values of the test set based on the fitted SVM Model
        scores = t_classifier.fit(self.train[:, 0:57], self.train[:, 57:58]).decision_function(self.test[:, 0:57])

        # Exp 1 using 200 evenly spaced threshold and score from test
        if one:
            # Precision, Accuracy & Recall Report with Threshold of zero
            accuracy, precision, recall, fpr = self.precision_recall_calc(scores, self.test[:, 57:58], 0)
            print "Using C = " + str(self.c) + " and a threshold of 0" + "\n"
            print "Accuracy: " + str(accuracy) + "\n"
            print "Precision: " + str(precision) + "\n"
            print "Recall: " + str(recall) + "\n"
            print "False Positive Rate: " + str(fpr) + "\n"

            # Initialise array for x and y axis, FPR and TPR(Recall) respectively
            tpr_array = []
            fpr_array = []
            # Generate 200 evenly spaced numbers between -1 amd 1
            even_thresh = np.linspace(start=-1, stop=1, num=200)
            for ii in range(len(even_thresh)):
                _, _, t, f = self.precision_recall_calc(scores, self.test[:, 57:58], even_thresh[ii])
                tpr_array.append(t)
                fpr_array.append(f)

            # Plotting ROC curve
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.axis([0, 1, 0, 1])
            plt.plot(fpr_array, tpr_array)
            plt.title("ROC Curve for 200 evenly spaced thresholds")
            plt.show()

        # Exp 2
        if two:
            # Experiment Two
            acc_array = []
            weight_vector = t_classifier.coef_          # weight vector of final SVM Model
            # print weight_vector
            for m in range(2, 58):                      # loop with m values from 2 to 57
                # returns the index of m largest
                indices_2 = heapq.nlargest(m, xrange(len(weight_vector[0])), abs(weight_vector[0]).take)
                # Initialise linear SVC using c*
                classify_2 = svm.SVC(C=self.c, kernel='linear', shrinking=True)
                # Fit the model using m features
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # This creates a SVM model using the features with the appropriate m features
                classify_2.fit(self.test[:, indices_2], self.test[:, 57:58])
                predict_2 = classify_2.predict(self.train[:, indices_2])             # Predict using just m features
                match_2 = self.match_values(predict_2, self.train[:, 57:58])         # Calc Accuracy
                # print str(match) + " / " + str(len(t_matrix))
                acc_2 = float(match_2)/len(self.train)
                acc_array.append(acc_2)
                # print str(acc) + " " + str(m)
                # print acc_array

            # Plotting Accuracy Vs M
            plt.ylabel('Accuracy')
            plt.xlabel('no of M features')
            plt.ylim(0,1.0)
            plt.plot(range(2,58), acc_array)
            plt.title("(Highest Weight Selection) Accuracy vs No of features with highest weights")
            plt.show()

            print indices_2
            print weight_vector

        # Exp 3
        if three:
            # Experiment Two
            acc_array = []
            for m in range(2, 58):                      # loop with m values from 2 to 57
                # random m indices
                indices_2 = random.sample(range(0, 57), m)
                # Initialise linear SVC using c*
                classify_2 = svm.SVC(C=self.c, kernel='linear', shrinking=True)
                # Fit the model using m features
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                # This creates a SVM model using the features with the appropriate m features
                classify_2.fit(self.train[:, indices_2], self.train[:, 57:58])
                predict_2 = classify_2.predict(self.test[:, indices_2])             # Predict using just m features
                match_2 = self.match_values(predict_2, self.test[:, 57:58])         # Calc Accuracy
                # print str(match) + " / " + str(len(t_matrix))
                acc_2 = float(match_2)/len(self.test)
                acc_array.append(acc_2)
                # print str(acc) + " " + str(m)

            # Plotting Accuracy Vs M
            plt.ylabel('Accuracy')
            plt.xlabel('no of M features')
            plt.ylim(0,1.0)
            plt.plot(range(2,58), acc_array)
            plt.title("(Random Weight Selection) Accuracy vs No of features with highest weights")
            plt.show()


if __name__ == "__main__":
    n = SVM()                                               # creating object for SVM
    n2 = SVM()
    n3 = SVM()
    n.experiments(one=False, two=True, three=False)           # set experiment number to true to see result
    n2.experiments(one=False, two=True, three=False)
    n3.experiments(one=False, two=True, three=False)


