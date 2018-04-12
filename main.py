import csv
import math
import random
import sys
import pandas
import numpy as np
import copy
import time

# global that will be the dataset to use
data_set = None
# global that will be the feature set of the data
feature_set = None
# global that will be the class set of the data
class_set = None


#
# This function will take in scores and perform the sigmoid function and return it
#
def sigmoid(scores):
	# Preform sigmoid
	return 1 / (1 + np.exp(-scores))


#
# This is the function that will return the log likelihood of the target
#
def log_likelihood(features, target, weights):
	# Get the dot product of all the scores
	scores = np.dot(features, weights)
	# Preform the log likelihood function
	# print scores
	# for i in range(len(scores)):
	# 	if scores[i] > 709:
	# 		scores[i] = 0
	ll = np.sum( target*scores - np.log(1 + np.exp(scores)) )
	# Return log likelihood
	return ll


#
# This is the logisitc regression algorithm
#
def logistic_regression(features, target, num_steps, learning_rate, add_intercept = False):
	# This is an option if you wish to add the intercept
	if add_intercept:
		intercept = np.ones((features.shape[0], 1))
		features = np.hstack((intercept, features))
		
	# Get a list of weights and set them all to 0
	weights = np.zeros(features.shape[1])
	# For each step in the number of steps you wish to take
	for step in xrange(num_steps):
		# Get the scores by getting the dot product of the scores and the weights
		scores = np.dot(features, weights)
		# Get the prediction of the scores
		predictions = sigmoid(scores)
		print predictions
		time.sleep(1)
		# Update weights with gradient
		output_error_signal = target - predictions

		gradient = np.dot(features.T, output_error_signal)
		weights += learning_rate * gradient
		
		# Print log-likelihood every so often
		if step % 10000 == 0:
			print log_likelihood(features, target, weights)
		
	return weights


#
# This function will get the mean of the array passed in
#
def mean(arr):
	# Get the mean of the entire array of numbers that was passed in
	return sum(arr)/float(len(arr))


#
# This function will get the standard dieviation of the array passed in
#
def stdev(arr):
	# Get the standard deviation of the entire array that was passed in
	# Start by getting the variance
	variance = sum([pow(x-mean(arr),2) for x in arr])/float(len(arr)-1)
	# Then take the square root of the variance
	return math.sqrt(variance)


def summarize(data):
	# the "zip" function will give us an iterable for each row of the data so we can get the attributes
	#	for evey instance of data
	
	summaries = []
	count = 0
	for ele in data[0][:-1]:
		temp_arr = []
		for row in data:
			temp_arr.append(row[count])
		summaries.append((mean(temp_arr), stdev(temp_arr)))
		count += 1

	# Return the summaries
	return summaries


#
# This function will separate the data by each classifier
#
def separate_by_class(data):
	# Create an empty set for put the classes, for now only going to be 1 or 0
	classes = {}
	# For every row in the data
	for i in range(len(data)):
		# Get a reference to the row
		row = data[i]
		# If you have not seen the class before
		if (row[-1] not in classes):
			# Add the class to the classes set
			classes[row[-1]] = []
		# Add the row to the corresponding class in the classes set
		classes[row[-1]].append(row)
	# Return the separated set
	return classes


#
# This function will get the summaries of each class for the dataset
#
def summarize_by_class(data):
	# Separate all of the classes
	separated = separate_by_class(data)

	# Set an empty set
	summaries = {}
	# For each class
	for instances in separated:
		# Get the summary of that class
		summaries[instances] = summarize(separated[instances])
	return summaries


#
# This function will calculate the probability of the classifier
#
def calculate_probability(classifier, mean, stdev):
	# Calculate probability
	if (2*math.pow(stdev,2)) == 0:
		return 1
	# Gaussian Probablility
	exponent = math.exp(-(math.pow(classifier-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


#
# This function will calculate the probability of the classes based on summaries and the input
# vector
#
def calculate_class_probabilities(summaries, inputVector):
	# Set an empty list to house the probabilities
	probabilities = {}
	# For each class in the summaries list
	for classValue, classSummaries in summaries.iteritems():
		# Set the corresponding class value to be 1
		probabilities[classValue] = 1
		# For each class summary
		for i in range(len(classSummaries)):
			# Get the mean and standard deviation of the class summary
			mean, stdev = classSummaries[i]
			# Get a reference to the particular class 
			classifier = inputVector[i]
			# Get the probability based on that class and multiply it to the corresponding
			# class value
			probabilities[classValue] *= calculate_probability(classifier, mean, stdev)
	# Return all of the probabilities
	return probabilities


#
# This function will make a prediction of the input vector based on the summaries
#
def predict(summaries, inputVector):
	# Get the probability of all the classes
	probabilities = calculate_class_probabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	# For each of the probabilies get the the best
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel


#
# This function will get the predictions of the test set based on the summaries
#
def get_predictions(summaries, testSet):
	# Create an empty list that will house the predictions
	predictions = []
	# For all of the data in the test set
	for i in range(len(testSet)):
		# Get the result of the perdiction
		result = predict(summaries, testSet[i])
		# Add the result to the predictions list
		predictions.append(result)
	# Return all of the predictions
	return predictions


#
# This function will calculate the classification accuracy
#
def get_accuracy(testSet, predictions):
	# Set a variable to 0
	correct = 0
	# For each entry in test set
	for x in range(len(testSet)):
		# If you got the class correct
		if testSet[x][-1] == predictions[x]:
			# Add 1 to the correct variable
			correct += 1
	# Return the accuracy by dividing the number of correct guesses by the total
	# length of the test set. Multiply by 100 to get percent
	return (correct/float(len(testSet))) * 100.0


#
# This function will run the naive bayes algorithm and return the accuracy
#
def naive_bayes_algo(test_set, training_set):
	# Summarize the attributes
	summaries = summarize_by_class(training_set)
	# Get the predictions of the test set
	predictions = get_predictions(summaries, test_set)
	#Get the accuracy
	accuracy = get_accuracy(test_set, predictions)
	# print('Naive Bayes Accuracy: {0}%').format(accuracy)
	return accuracy


def read_csv(filepath):
	# Make a reference to the data set
	data = []
	# Make a reference to the attributes
	attributes = []
	# Make a counter variable
	counter = 0
	# Open the file at filepath
	with open(filepath) as tsv:
		# For each line in the file separated by commas
		for line in csv.reader(tsv, delimiter=","):
			# If you are on the first line
			if counter == 0:
				# Set the attributes variable
				attributes = line
				# Make the counter 1
				counter = 1
			# Append to data set
			else:
				data.append(tuple(line))
	# Return a tuple with the information we read
	return (attributes, data)


#
# This function will give back a fold in the cross fold validation
#
def cross_fold_sets(data, k, K):
	# Randomize the data
	random.shuffle(data)
	# Get the training set of the fold
	training_set = [x for i, x in enumerate(data) if i % K != k]
	# Get the validation set of the fold
	validation_set = [x for i, x in enumerate(data) if i % K == k]
	# Return the sets
	return (training_set, validation_set)


if __name__ == "__main__":
	# Read in the file passed in by the command line when script started
	info = read_csv(sys.argv[1])

	data = cross_fold_sets(info[1], 1, 5)
	train_data = data[0]
	test_data = data[1]


	train_feats = []
	train_class = []
	for row in train_data:
		c = row[-1]
		row = [float(i) for i in row[:-1]]
		# row.append(c)
		train_class.append(c)
		train_feats.append(row)

	test_feats = []
	test_class = []
	for row in test_data:
		c = row[-1]
		row = [float(i) for i in row[:-1]]
		# row.append(c)
		test_class.append(c)
		test_feats.append(row)



	feats = np.array(train_feats, dtype=np.float64)
	clas = np.array(train_class,  dtype=np.float64)

	test_f = np.array(test_feats,  dtype=np.float64)
	test_c = np.array(test_class,  dtype=np.float64)

	np.random.seed(12)
	num_observations = 5000

	# x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
	# x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

	# simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
	# simulated_labels = np.hstack((np.zeros(num_observations), np.ones(num_observations)))

	weights = logistic_regression(feats, clas, num_steps = 30000, learning_rate = 5e-5, add_intercept=True)
	# print feats
	data_with_intercept = np.hstack((np.ones((test_f.shape[0], 1)), test_f))
	# print data_with_intercept

	final_scores = np.dot(data_with_intercept, weights)
	print final_scores
	preds = np.round(sigmoid(final_scores))
	print preds

	# print preds
	print 'Accuracy from scratch: {0}'.format((preds == test_c).sum().astype(float) / len(preds))
	# print removed_class_col
	# for i in range(5):
	# 	data = cross_fold_sets(removed_class_col, i, 5)

	# 	print naive_bayes_algo(data[1], data[0])