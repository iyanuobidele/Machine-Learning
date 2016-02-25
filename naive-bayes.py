# Homework 4
# Classification of emails into Spam and No Spam using Gaussian Naive Bayes

import numpy as np
import csv
import math
from sklearn import metrics


class GNaiveBayes(object):
    def __init__(self):
        # Initialise some self variables
        self.model = {}             #  {feature index: [+ve mean, +ve stddev, -ve mean, -ve stddev], ...., }
        self.PofSpam = 0
        self.PofNotSpam = 0
        self.test = []
        self.train = []

    def create_model(self):
        # This function would be responsible for creating the probabilistic model of the training set

        # Training Data in 2301 by 58 including target
        with open(r'C:\Users\Iyanu\Desktop\Python\ML\data\4\training.data','r') as dest_f:
            data_iter = csv.reader(dest_f,
                                   delimiter=',',
                                   quotechar='"')
            data = [data for data in data_iter]
        self.train = np.asarray(data, dtype=float)

        # Test Data in 2300 by 58 including target
        with open(r'C:\Users\Iyanu\Desktop\Python\ML\data\4\test.data','r') as tdest_f:
            tdata_iter = csv.reader(tdest_f,
                                    delimiter=',',
                                    quotechar='"')
            tdata = [tdata for tdata in tdata_iter]
        self.test = np.asarray(tdata, dtype=float)

        # For each feature, we need to compute the mean, and stddev of each positive and negative label
        # we need an outer loop to iterate over all columns except the label
        pos = []
        neg = []
        for i in range(0,57): # loop for every feature (Index 0-56) we don't need to create a model for the label
            pos_mean = 0; pos_stddev = 0; neg_mean = 0; neg_stddev = 0
            for ii in range(len(self.train.T[i])):  # loop each value inside a column.
                # find feature attributes belonging to the spam and no spam class
                if self.train.T[-1][ii] == 1:
                    pos.append(self.train.T[i][ii])     # positive class features (Spam)
                if self.train.T[-1][ii] == 0:
                    neg.append(self.train.T[i][ii])     # negative class features (No Spam)

            # compute the means and stddev
            print len(pos), len(neg)
            pos_mean = np.mean(pos)                         # spam mean calculation
            if math.isnan(np.std(pos)) or np.std(pos) == 0: # check if spam stddev is nan or 0
                pos_stddev = 0.000000000001
            else:
                pos_stddev = np.std(pos)
            neg_mean = np.mean(neg)                         # notSpam mean calculation
            if math.isnan(np.std(neg)) or np.std(neg) == 0: # check if notSpam stddev is nan or 0
                neg_stddev = 0.000000000001
            else:
                neg_stddev = np.std(neg)

            # positive and negative probabilities
            if self.PofNotSpam == 0:                                    # to make this is only done once
                self.PofSpam = float(len(pos))/len(self.train)
                self.PofNotSpam = float(len(neg))/len(self.train)

            # empty arrays before next loop
            pos = []
            neg = []
            # insert into model dictionary
            self.model[i] = [pos_mean, pos_stddev, neg_mean, neg_stddev]
        print self.model
        self.prediction()

    def prediction(self):
        # This function will be used to classify all instances of the test set.
        prob_of_positives = []
        prob_of_negatives = []
        predicted = []
        counter = 0
        for a in range(len(self.test)):
            for aa in range(len(self.test[a][:-1])):
                # for each feature value, we need to calculate the p(fi = x | +) and p(fi = x | -)

                # p(fi = x | +)
                temp1 = float(1)/(np.sqrt(2*np.pi)*self.model[aa][1])
                temp2 = np.e**-((self.test[a][aa] - self.model[aa][0])**2 / (2 * self.model[aa][1]**2))
                temp_pos_prob = temp1 * temp2
                prob_of_positives.append(temp_pos_prob)
                team_pos_prob = 0
                temp1 = 0
                temp2 = 0

                # p(fi = x | -)
                temp4 = float(1)/(np.sqrt(2*np.pi)*self.model[aa][3])
                temp5 = np.e**-((self.test[a][aa] - self.model[aa][2])**2 / (2 * self.model[aa][3]**2))
                temp_neg_prob = temp4 * temp5
                prob_of_negatives.append(temp_neg_prob)
                temp_neg_prob = 0
                temp4 = 0
                temp5 = 0

            print len(prob_of_negatives), len(prob_of_positives)

            # positive
            positive = np.log(self.PofSpam) * np.log(np.prod(np.array(prob_of_positives)))

            # negative
            negative = np.log(self.PofNotSpam) * np.log(np.prod(np.array(prob_of_negatives)))

            print positive, negative, np.argmax([positive, negative])

            # accuracy
            classification = np.argmax([positive, negative])
            predicted.append(classification)

            if classification == self.test[a][-1]:
                counter += 1

            prob_of_negatives = []
            prob_of_positives = []

        acc = float(counter)/len(self.test)

        print("Classification report for classifier \n%s\n" % (metrics.classification_report(self.test.T[-1], predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(self.test.T[-1], predicted))
        print "Accuracy: " + str(acc)
        # print acc

if __name__ == "__main__":
    n = GNaiveBayes()    # creating an object
    n.create_model()