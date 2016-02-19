"""
+++++++++++++++++++++++++++++++++++++++++
+ Home work 1 -                         +
+ Perceptron Learning Algorithm &       +
+ All pairs Classification Method       +
+++++++++++++++++++++++++++++++++++++++++
"""

"""
Import Libraries necessary for computation. 
"""
import random
from random import choice
import numpy as np
import matplotlib
import pandas as pd
import json
# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap


# Since all 325 perceptrons have the same functionality of classification, it makes sense to generalize the different functions in a class #
class Perceptron(object):
	"""
	Parameters:
	1. eta: Learning rate
	2. n_iter: Number of epoch
	3. p_type: Perceptron identifier #To differentiate all 325 perceptron, since that will determine the training example needed for training.
	"""

	"""
	Attribute & Data Structure:
	- Weight after training
	- No of misclassifications in each epoch
	- Target/Expected Value: +1 or -1
	- Translation value: In the perceptron a vs b; +1 will mean a and  -1 will mean b.
	- number of epoch
	Global data structure that can be accessed by every object of the class to store attributes training
	"""

	def __init__(self, eta=0.2):
		self.eta = eta
		self.final_1 = {}
		self.predict = {}
		self.df = pd.read_csv(r'C:\Users\Iyanu\Desktop\Python\ML\data\training.data')
		self.d = pd.read_csv(r'C:\Users\Iyanu\Desktop\Python\ML\data\test.data')
		with open(r'C:\Users\Iyanu\Desktop\Python\ML\data\final.json','r+') as datafile:
			self.test = json.load(datafile)
	
	# @staticmethod
	def lookup(self, x):

		# perceptron will be used internally as numbers, this is needed to interpret the numbers.
		alpha = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "J", 10: "K", 11: "L",
				 12: "M", 13: "N", 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W",
				 23: "X", 24: "Y", 25: "Z"}
		return alpha[x]

	def frange(self, x, y, jump):
		while x < y:
			yield x
			x += jump
	
	def rand_weight(self,n):
		arr = []
		for i in range(n):
			arr.append(round(random.uniform(-1,1),2))
		return arr

	def signum(self, values):
		"""
		Parameters passes:
		values - [[target, decision-value, (w-weight-vector), b-bias, x-input]]
		"""
		# Declare some integer variables for statistics
		results = {}
		total_eg1 = 0    # total number of class 1 received for classification
		total_eg2 = 0  	 # total number of class 2 received for classification
		correct_eg1 = 0  # correctly classified class 1
		correct_eg2 = 0  # correctly classified class 2
		error_eg1 = 0
		error_eg2 = 0
		total = len(values)  # Total number of [target,decision-Value] pairs received for classification. For statistics.

		for i in range(0, len(values)):
			if values[i][1][1] == values[i][0][1][0]:
				total_eg1 += 1  												# Total number of [class 1,decision-Value] pairs received for classification
			if values[i][1][1] == values[i][0][1][1]:
				total_eg2 += 1  												# Total number of [class 2,decision-Value] pairs received for classification

		for i in range(0, len(values)):
			if values[i][1][1] == values[i][0][1][0] and values[i][2][1] >= 0:	# increases counter if target is class_1 and d-v is >=0
				correct_eg1 += 1
			if values[i][1][1] == values[i][0][1][0] and values[i][2][1] < 0:	# increases counter if target is class_1 and d-v is < 0
				error_eg1 += 1
			if values[i][1][1] == values[i][0][1][1] and values[i][2][1] < 0:	# increases counter if target is class_2 and d-v is < 0 
				correct_eg2 += 1
			if values[i][1][1] == values[i][0][1][1] and values[i][2][1] >= 0:	# increases counter if target is class_2 and d-v is >=0
				error_eg2 += 1

		correct = correct_eg1 + correct_eg2
		ac = float(correct) / total

		# shows statistics, of training example and weight used to achieve the stated accuracy
		results = {"perceptron": values[0][0][1][0] + " VS " + values[0][0][1][1], "accuracy": ac, "total-1": total_eg1,
				   "total-2": total_eg2, "correct-1": correct_eg1, "correct-2": correct_eg2, "weight": values[0][3][1],
				   "bias": values[0][4][1]}

		return results

	def training(self, n_iter):
		#########################################################################################
		# Depending on initial accuracy, we might miraculously get it right, we should stop.    #
		# In other cases, we use the number of iterations given to us for learning.             #
		# Better yet, it can use both, using the number of iterations and --                    #
		# should stop, if we get a good accuracy.                                               #
		#########################################################################################

		# values = []
		# final = {}
		# Datafile for training

		# start training of all 325 perceptrons #this would simulate k(k-1)/2 perceptrons
		for row in range(0,25):
			for col in range(row+1, 26):
				
				final = {}
				# generate random weights and bias starting w1...w16
				bias = round(random.uniform(-1, 1), 2)
				w = self.rand_weight(16)
				values = []
				
				# To know what training samples to use. I need to know which perceptrons we are on. e.g 24,25 - Y,Z
				eg1 = self.lookup(row)
				eg2 = self.lookup(col)
				
				"""
				a = self.df.loc[self.df['target'] == eg1]
				b = self.df.loc[self.df['target'] == eg2]
				new_a = np.array(a)
				new_b = np.array(b)
				"""
				new_df = self.df.loc[self.df['target'] == eg1].append(self.df.loc[self.df['target'] == eg2])  # combine both groups of training example into one data file, makes it easy to try to compute all the dot product.
				
				new = np.array(new_df)
				np.random.shuffle(new)			# shuffle ndarray
				
				length = len(new)
				for i in range(0, length):  																				# loop through all the training examples
					temp = np.dot(new[i][1:17],w) + bias  																	# dot product of all training examples and weight + bias
					values.append([['p', [eg1, eg2]], ['target', new[i][0]], ['dv', temp], ['weights', w],['bias', bias]])  # appending all the necessary details, needed for classification
						
				sgn = self.signum(values)  # calling the signum function that returns the classification detail as a dictionary
				
				print "Perceptron "+ sgn['perceptron']
				print "Using random w's, the initial accuracy of perceptron " + sgn['perceptron'] + " is: " + str(sgn['accuracy'])
				
				
				if sgn['accuracy'] == 1 or sgn['accuracy'] > self.test[sgn['perceptron']]['a']:																				# if with first random w we get an acc of 1, we need to save.
					self.test[sgn['perceptron']] = {"w":sgn['weight'],"b":sgn['bias'],"a":sgn['accuracy']}
					with open(r'C:\Users\Iyanu\Desktop\Python\ML\data\final.json', 'w') as f:
						json.dump(self.test, f)
					print "Perceptron " + sgn['perceptron'] + " has an accuracy of: " + str(sgn['accuracy']) + " using weight and bias: " + str(sgn['weight'])
				else:
					sgn['weight'].insert(0,sgn['bias'])
					self.learning(n_iter,sgn['perceptron'],sgn['weight'],sgn['accuracy'],eg1,eg2) # calls the learning function that updates the weight and runs the signum again.
		
		print "All Done!"
		self.training_acc()
		#with open(r'C:\Users\Iyanu\Desktop\Python\ML\data\final.json', 'w') as f:
		#	json.dump(self.test, f)
		
	def learning(self,n,p,w_b,a,class_1,class_2):
		"""
		Parameters:
		n: number of iterations
		p: perceptron
		w_b: w0..w17
		a: accuracy
		class_1: Alphabet 1. e.g A in A vs D perceptron
		class_2: Alphabet 2. e.g D in example above. 
		"""
		#############################################################################################################################################
		# Here we basically update the weights using eta and the training examples related to the perceptron, and re run signum					    #
		# We do this depending on the amount of stipulated iterations, for each perceptron															#
		#############################################################################################################################################
		for count in range(n):
			print "Epoch " + str(count+1)
			new_df = self.df.loc[self.df['target'] == class_1].append(self.df.loc[self.df['target'] == class_2])
			tr_new = np.array(new_df)  																					# reindex(shuffle) the datafile
			np.random.shuffle(tr_new)
			for ii in range(len(tr_new)):  																				# loop to choose from the training example
				if tr_new[ii][0] == class_1:  																				# condition to handle all w0
					target = 1
				if tr_new[ii][0] == class_2:
					target = -1
				for i in range(17):  																					# To apply eta on all weights including bias
					if i == 0:
						x = 1																							# x0 for w0
					if i != 0:
						x = tr_new[ii][i]  																					# x1......x16
					w_b[i] += round((self.eta*x*target),2)																# Learning Algorithm
				
				new_values = []
				for o in range(0, len(tr_new)):  																		# loop through all the training examples
					temp = np.dot(tr_new[o][1:17],w_b[1:17]) + w_b[0]  																	# dot product of all training examples and weight + bias
					new_values.append([['p', [class_1, class_2]], ['target', tr_new[o][0]], ['dv', temp], ['weights', w_b[1:17]],['bias', w_b[0]]])  # appending all the necessary details, needed for classification
				new_sgn = self.signum(new_values)
								
				if new_sgn['accuracy'] > 0.8 and new_sgn['accuracy'] > self.test[new_sgn['perceptron']]['a']:
					self.test[p] = {"w":new_sgn['weight'],"b":new_sgn['bias'],"a":new_sgn['accuracy']}
					print p +" will be saved with an accuracy of: "+ str(new_sgn['accuracy']) + "\n"
					with open(r'C:\Users\Iyanu\Desktop\Python\ML\data\final.json', 'w') as f:
						json.dump(self.test, f)
					break																						# no need to go through training examples, since sgn has improved to what we want
				else:
					new_values = []																				# else
			if new_sgn['accuracy'] > 0.8 and new_sgn['accuracy'] > self.test[new_sgn['perceptron']]['a']:
				self.test[p] = {"w":new_sgn['weight'],"b":new_sgn['bias'],"a":new_sgn['accuracy']}
				with open(r'C:\Users\Iyanu\Desktop\Python\ML\data\final.json', 'w') as f:
					json.dump(self.test, f)
				print p +" will be saved with an accuracy of: "+ str(new_sgn['accuracy']) + "\n"
				break																								# break out of this function to the next perceptron
			else:
				w_b = self.rand_weight(17)																		# the last known weights should be base weights, by SGD standard.
		
	
	def training_acc(self):
		a = 0
		for i in self.test.keys():
			a += self.test[i]['a']
		
		accuracy = float(a)/len(self.test)
		print "Training accuracy is: " + str(accuracy)
	
	###################################################################
	# All pairs classification method is used for prediction		  #
	###################################################################
	def prediction(self):
		tr_new = np.array(self.d)																				# transferring all data into an ndarray
		counter = 0
		target = []
		prediction = []
		for o in range(0, len(tr_new)):  																		# loop through all the training examples
			# start testing w/ all 325 perceptrons 
			a = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
			for row in range(0,25):
				for col in range(row+1, 26):																	# loop through all perceptrons
					eg1 = self.lookup(row)
					eg2 = self.lookup(col)
					p = eg1+ " VS " +eg2
					# print p											# perceptron name						
					# print tr_new[0][0]								# target
					temp = np.dot(tr_new[o][1:17],self.test[p]['w']) + self.test[p]['b']						# dot product of all training examples and weight + bias
					
					if temp >= 0:																				# condition for classification
						a[row]+=1																				# this is a vote for C in C VS B. if decision value is >= 0
					else:
						a[col]+=1																				# this is a vote for B in the e.g above
				
			predict = self.lookup(a.index(max(a)))																# Gets the index with the most votes and gets the alphabet name
			# print tr_new[o][0] +" is "+ predict
			target.append(tr_new[o][0])																			# append all the actual into an array
			prediction.append(predict)																			# append all the predicted into an array
			if predict == tr_new[o][0]:
				counter+=1																						# counter for number of times we predicted right
		
		accuracy = float(counter)/len(tr_new)																	# accuracy calculation
		print " Accuracy is: "+ str(accuracy) +"\n"
		
		###################
		#Confusion Matrix #
		###################
		x = pd.Series(target, name='Actual')
		y = pd.Series(prediction, name='Predicted')
		confusion = pd.crosstab(x,y,rownames=['Actual'], colnames=['Predicted'])
		with pd.option_context('display.max_rows', 27, 'display.max_columns', 27):
			print confusion

		
p = Perceptron(eta=0.2)
#p.training(40)				# for perceptron training
#p.training_acc()			# for perceptron training accuracy
p.prediction()				# for perceptron prediction.