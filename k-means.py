# Homework 5
# Classification of optdigits using supervised and unsupervised learning - K Means Algorithm

import numpy as np
import csv
from sklearn import metrics
import matplotlib.pyplot as plot


class KMeans(object):
    def __init__(self, k=10):
        self.train = []             # Training set
        self.test = []              # Test set
        self.k = k                  # Amount of clusters/centroids
        self.colLength = 64         # Dimension of each centroid
        self.centroids = []         # array storing centroids

    def initial_centroids(self):
        # Generate initial random centroids at the beginning of each run
        # self.centroids = np.random.uniform(low=0, high=16, size=(self.k, self.colLength))
        return np.random.random_integers(low=0, high=16, size=(self.k, self.colLength))

    def entropy_check(self, arr):
        """
        :param arr:
        :return: Bool: True or False
        """
        temp = arr[0]
        for i in range(1,len(arr)):
            if temp != arr[i]:
                return False
        return True

    def create_model(self):
        # Set the data model in a np array
        # Training Data including target
        with open(r'C:\Users\Iyanu\Desktop\Python\ML\data\5\optdigits\optdigits.train','r') as dest_f:
            data_iter = csv.reader(dest_f,
                                   delimiter=',',
                                   quotechar='"')
            data = [data for data in data_iter]
        self.train = np.asarray(data, dtype=float)

        # Test Data including target
        with open(r'C:\Users\Iyanu\Desktop\Python\ML\data\5\optdigits\optdigits.test','r') as tdest_f:
            tdata_iter = csv.reader(tdest_f,
                                    delimiter=',',
                                    quotechar='"')
            tdata = [tdata for tdata in tdata_iter]
        self.test = np.asarray(tdata, dtype=float)

    def eucl_dist(self, centroids):
        # Responsible for calculating the euclidean distance between data points and centroids
        temp = 0
        temp_array = []
        clustering = dict((el,[]) for el in range(self.k))
        for a in range(len(self.train)):                                        # for each data point
            for aa in range(len(centroids)):                                    # iterate through all centroids
                for aaa in range(len(self.train[a]) - 1):                           # pick each value of the data point
                    temp += np.square(self.train[a][aaa] - centroids[aa][aaa])  # (x-m)*(x-m) for each value
                temp_array.append(np.sqrt(temp))                                # temporarily append the sqrt of the sum
                temp = 0
            closest_c = np.argmin(temp_array)           # index of the smallest value represents the closest centroid
            # create my clustering data structure. {centroid_1 :[data-points],...,centroid_10:[data-points]}
            clustering[closest_c].append(a)
            temp_array = []

        return clustering

    def new_mean(self, clustering, centroids):
        # responsible for finding the mean of clusters
        temp_c_array = []
        new_centroids = []
        count = 0
        for i in clustering.itervalues():               # iterate through the list of clusters
            if len(i) > 1:                                       # if cluster not empty
                cluster = self.train[i].T               # store the transpose of the cluster data-points
                for ii in range(len(cluster) - 1):      # iterate through the list leaving out the labels
                    temp_c_array.append(np.mean(cluster[ii]))   # store temporarily the mean of each transposed column
                new_centroids.append(temp_c_array)              # updated centroid is appended
                temp_c_array = []
            elif len(i) == 1:                                   # if cluster has just one value then that's new mean
                new_centroids.append(centroids[count])
            else:
                new_centroids.append(centroids[count])          # if cluster is empty retain old centroid values
            count += 1
        return new_centroids

    def calc_sse(self, clustering, centroids):
        # responsible for calculating the Sum Squared Error of clustering
        sse = 0
        for i in range(len(self.train)):    # for each data-point
            for cluster_i, datapoints in clustering.iteritems():        # search for its centroid
                if i in datapoints:
                    c = cluster_i                                       # centroid/index
            for n in range(self.colLength):                             # iterate through vector leaving out the label
                sse += np.square(self.train[i][n] - centroids[c][n])    # (x1-m1)*(x1-m1)+,.,+(x64-m64)*(x64-m64)
        # self.sse = sse
        # print "SSE :" + str(sse)
        return sse

    def calc_sss(self, centroids):
        # responsible for calculating the Sum Squared Separation of centroids
        sss = 0
        # create all distinct pairs of centroids
        for row in range(0, self.k - 1):
            for col in range(row+1, self.k):
                for i in range(self.colLength):    # iterate through each centroid
                    sss += np.square(centroids[row][i] - centroids[col][i])
        # self.sss = sss
        # print "SSS :" + str(sss)
        return sss

    def mean_entropy(self, clustering):
        count = 0                                   # initial count 0
        mean_e = 0                                  # initial mean
        entropy_of_cluster = 0                      # initial entropy = 0
        actual_classes = self.train.T[-1]           # classes for all training examples
        for i in clustering.itervalues():           # for every cluster
            cluster_labels = actual_classes[i]      # Cluster classes
            length = len(cluster_labels)             # get the length of examples in the cluster
            # Entropy Calculation
            # if cluster contains only one label then entropy is 0
            if length > 0:
                if self.entropy_check(cluster_labels):
                    entropy_of_cluster = 0
                else:
                    for ii in clustering.iterkeys():
                        if ii in cluster_labels:
                            temp = float(sum(cluster_labels == ii)) / length
                            entropy_of_cluster += (temp * np.log2(temp))
                # print "Entropy of Cluster " + str(count) + ":" + " " + str(entropy_of_cluster)
                count += 1
                mean_e += float(length)/len(self.train) * -entropy_of_cluster
                entropy_of_cluster = 0
        # self.mean_e = mean_e
        # print "Mean Entropy: " + str(mean_e)
        return mean_e

    def training(self, iteration=5):
        count = 0
        final = []                                      # Info: [SSE, SSS, mean_entropy, [centroids],[clustering]]
        self.create_model()                             # training model
        while count < iteration:
            # K-means Algorithm
            inner_count = 0
            print "Run " + str(count + 1)
            while True:
                # iteration should use last mean (if there's one)
                if self.centroids:
                    temp_centroids = self.centroids
                else:
                    temp_centroids = self.initial_centroids()           # initial random centroids
                temp_clustering = self.eucl_dist(temp_centroids)        # Calculate Euclidean Distance
                temp_new_centroids = self.new_mean(temp_clustering, temp_centroids)     # update the centroids as the mean of cluster
                temp_sse = self.calc_sse(temp_clustering, temp_new_centroids)       # calculate SSE
                temp_sss = self.calc_sss(temp_new_centroids)                        # calculate SSS
                temp_m_e = self.mean_entropy(temp_clustering)                       # calculate mean entropy
                print "Iteration " + str(inner_count+1) + " Sum Squared Error: " + str(temp_sse)
                print "Iteration " + str(inner_count+1) + " Sum Squared Separation: " + str(temp_sss)
                print "Iteration " + str(inner_count+1) + " Mean Entropy: " + str(temp_m_e) + "\n"
                # Stop when cluster centers stop changing or if algorithm is stuck in an oscillation
                if count == 100 or np.array_equal(temp_centroids,temp_new_centroids):
                    print "End of this run, information saved"
                    if not final:
                        final = [temp_sse, temp_sss, temp_m_e, temp_centroids, temp_clustering]
                    else:
                        if final[0] > temp_sse:        # if SSE of current run is smaller update info
                            final = [temp_sse, temp_sss, temp_m_e, temp_centroids, temp_clustering]
                    break
                inner_count += 1
                self.centroids = temp_new_centroids # next iteration should make use of new mean from last iteration
            count += 1
            self.centroids = []
        return final
        # print self.final

    def prediction(self, final):
        temp = 0
        temp_array = []
        # analysing training clustering
        training_c = final[4]
        maximum = 0
        dicto = {}
        # some print jobs
        print "\nBest SSE: " + str(final[0])
        print "Its SSS: " + str(final[1])
        print "Its Mean Entropy: " + str(final[2])
        max_class = 0
        # associating cluster centers to most frequent class in its cluster
        for cluster, datapoints in training_c.iteritems():      # iterate through the clustering
            datapoint_classes = self.train.T[-1][datapoints]    # cluster actual classes
            for ii in training_c.iterkeys():                    # search through cluster for possible classes
                if ii in datapoint_classes:
                    if sum(datapoint_classes == ii) > maximum:  # cluster center should be associated w/ most freq class
                        maximum = sum(datapoint_classes == ii)
                        max_class = ii
            print "Cluster: " + str(cluster) + " is for " + str(max_class)
            dicto[cluster] = max_class
            maximum = 0

        # analysing test clustering
        # clustering = dict((el,[]) for el in range(self.k))
        centroids = final[3]
        counter = 0
        predicted = []
        for a in range(len(self.test)):                                        # for each data point
            for aa in range(len(centroids)):                                    # iterate through all centroids
                for aaa in range(len(self.test[a]) - 1):                           # pick each value of the data point
                    temp += np.square(self.test[a][aaa] - centroids[aa][aaa])  # (x-m)*(x-m) for each value
                temp_array.append(np.sqrt(temp))                                # temporarily append the sqrt of the sum
                temp = 0
            closest_c = np.argmin(temp_array)           # index of the smallest value represents the closest centroid
            if dicto[closest_c] == self.test[a][-1]:
                counter += 1
            predicted.append(dicto[closest_c])
            temp_array = []

        print("Classification report for classifier \n%s\n" % (metrics.classification_report(self.test.T[-1], predicted)))
        print("Confusion matrix:\n%s" % metrics.confusion_matrix(self.test.T[-1], predicted))
        print "Accuracy: " + str(float(counter) / len(self.test)) + "\n\n"

        # Visualizing cluster centers
        for ni in range(len(centroids)):
            plot.imshow(np.reshape(centroids[ni],(8,8)), interpolation="gaussian")
            plot.show()


if __name__ == "__main__":
    n = KMeans(k=30)    # creating an object
    final = n.training(iteration=5)
    n.prediction(final)