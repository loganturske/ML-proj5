import csv
import math
import random
import sys
import pandas as pd
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
# This function will calculate the softmax for the array passed in
#
def softmax(inputs):
	# Preform the softmax function and return the values
	return np.exp(inputs) / float(sum(np.exp(inputs)))


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
		# predictions = softmax(scores)

		# Update weights with gradient
		output_error_signal = target - predictions

		gradient = np.dot(features.T, output_error_signal)
		weights += learning_rate * gradient
		
		# Print log-likelihood every so often
		# if step % 10000 == 0:
		# 	print log_likelihood(features, target, weights)
		
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
	# Check for zero
	len_of_arr = float(len(arr)-1)
	if len_of_arr == 0:
		return 0
	# Start by getting the variance
	variance = sum([pow(x-mean(arr),2) for x in arr])/(float(len(arr)))
	# Then take the square root of the variance
	return math.sqrt(variance)


#
# This function will get all of the summaries of the mean and stddev of the data
#
def summarize(data):
	# Create an empty list to house all of the summaries
	summaries = []
	# Create an iterable
	count = 0
	# For each feature
	for ele in data[0][:-1]:
		# Create a temp list
		temp_arr = []
		# For every row in the data
		for row in data:
			# Append the row to the temp array
			temp_arr.append(row[count])
		# Add the mean and sdtdev to the summaries array
		summaries.append((mean(temp_arr), stdev(temp_arr)))
		# Increment the counter
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
	# Set running variables for best you have seen so far
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


print_once = 0
#
# This function will run the naive bayes algorithm and return the accuracy
#
def naive_bayes_algo(test_set, training_set):
	# Summarize the attributes
	summaries = summarize_by_class(training_set)
	# The assignment says to only print one set of weights out
	global print_once
	# If you havent printed yet
	if print_once == 0:
		# Print the summaries
		print summaries
		# Set the print_once global to 1
		print_once = 1
	# Get the predictions of the test set
	predictions = get_predictions(summaries, test_set)
	#Get the accuracy
	accuracy = get_accuracy(test_set, predictions)
	# Return accuracy
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

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

if __name__ == "__main__":
	# Read in the file passed in by the command line when script started
	info = read_csv(sys.argv[1])

	from sklearn import linear_model
	from sklearn import metrics
	from sklearn.cross_validation import train_test_split

	# glass_data_headers = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "glass-type"]
	# glass_data = pd.read_csv("glass.csv", names=glass_data_headers)

	# glass_data_headers = ["sepal length", "sepal width", "petal length", "petal width","class"]
	# glass_data = pd.read_csv("iris.csv", names=glass_data_headers)

	glass_data_headers = ["date", "plant-stand", "precip", "temp", "hail", "crop-hist", "area-damaged", \
	 "severity", "seed-tmt", "germination", "plant-growth", "leaves", "leafspots-halo", "leafspots-marg", "leafspot-size", \
	 "leaf-shread", "leaf-malf", "leaf-mild", "stem", "lodging", "stem-cankers", "canker-lesion", \
	 "fruiting-bodies", "external decay", "mycelium", "int-discolor", "sclerotia", "fruit-pods", "fruit spots", \
	 "seed", "mold-growth", "seed-discolor", "seed-size", "shriveling", "roots", "class"]
	glass_data = pd.read_csv("soybean.csv", names=glass_data_headers)	

	train_x, test_x, train_y, test_y = train_test_split(glass_data[glass_data_headers[:-1]], glass_data[glass_data_headers[-1]], train_size=0.7)
	# Train multi-class logistic regression model
	lr = linear_model.LogisticRegression()
	lr.fit(train_x, train_y)
	print lr.coef_[0]

	print "Logistic regression Accuracy :: ", metrics.accuracy_score(train_y, lr.predict(train_x))

	time.sleep(2)
	# data = cross_fold_sets(info[1], 1, 5)
	# train_data = data[0]
	# test_data = data[1]


	# train_feats = []
	# train_class = []
	# for row in train_data:
	# 	c = row[-1]
	# 	row = [float(i) for i in row[:-1]]
	# 	# row.append(c)
	# 	train_class.append(c)
	# 	train_feats.append(row)


	# test_feats = []
	# test_class = []
	# for row in test_data:
	# 	c = row[-1]
	# 	row = [float(i) for i in row[:-1]]
	# 	# row.append(c)
	# 	test_class.append(c)
	# 	test_feats.append(row)



	# feats = np.array(train_feats, dtype=np.float64)
	# clas = np.array(train_class,  dtype=np.float64)

	# test_f = np.array(test_feats,  dtype=np.float64)
	# test_c = np.array(test_class,  dtype=np.float64)


	# weights = logistic_regression(feats, clas, num_steps = 3000, learning_rate = 5e-5, add_intercept=True)
	# print weights
	# # print feats
	# data_with_intercept = np.hstack((np.ones((test_f.shape[0], 1)), test_f))
	# # print data_with_intercept

	# final_scores = np.dot(data_with_intercept, weights)
	# # print final_scores
	# preds = np.round(sigmoid(final_scores))
	# print preds
	removed_class_col = []
	for row in info[1]:
		c = row[-1]
		row = [float(i) for i in row[:-1]]
		row.append(c)
		removed_class_col.append(row)
	# print preds
	# print 'Accuracy from scratch: {0}'.format((preds == test_c).sum().astype(float) / len(preds))
	folds_acc = 0
	for i in range(5):
		data = cross_fold_sets(removed_class_col, i, 5)

		folds_acc += naive_bayes_algo(data[1], data[0])
	print "Naive Bayes Accuracy :: " + str(folds_acc/5)