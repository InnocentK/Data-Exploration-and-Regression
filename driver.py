## File: driver.py
## Date Created: 01/27/2019
## Author: Wambugu "Innocent" Kironji
## Class: ECE 580 - Introduction to Machine Learning
## Description:
##		....

import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn import linear_model

WINDOWS = "\\"
UNIX = "/"
OS = UNIX
DATASET = "." + OS + "auto_data" + OS + "imports-85.data"
PLOTS = "." + OS + "plots" + OS

RAW_DATA_DICT = {
	"symboling": 0,
	"normalized-losses": 1,
	"make": 2,
	"fuel-type": 3,
	"aspiration": 4,
	"num-of-doors": 5,
	"body-style": 6,
	"drive-wheels": 7,
	"engine-location": 8,
	"wheel-base": 9,
	"length": 10,
	"width": 11,
	"height": 12,
	"curb-weight": 13,
	"engine-type": 14,
	"num-of-cylinders": 15,
	"engine-size": 16,
	"fuel-system": 17,
	"bore": 18,
	"stroke": 19,
	"compression-ratio": 20,
	"horsepower": 21,
	"peak-rpm": 22,
	"city-mpg": 23,
	"highway-mpg": 24,
	"price": 25
}
DATA_DICT = {
	"wheel-base": 0,
	"length": 1,
	"width": 2,
	"height": 3,
	"curb-weight": 4,
	"engine-size": 5,
	"bore": 6,
	"stroke": 7,
	"compression-ratio": 8,
	"horsepower": 9,
	"peak-rpm": 10,
	"city-mpg": 11,
	"highway-mpg": 12
}
NUM_CONTIUOUS_VAR = 13
TOTAL_CARS = 205
TOTAL_ATRIB = 26

def load(filename = DATASET):
	file = open(filename, 'r')

	# Reading in all data as a list (strings for each line in the file)
	data_raw = file.readlines()

	# Seperating data into a 2D list (Rows are the different cars and columns are the attributes for the cars)
	data = [x.strip().split(',') for x in data_raw]

	file.close()
	print("Data loaded")
	return data

def clean_data(data):
	
	cleaned = []
	target = []

	# Getting the indecies for all the desired car attributes
	desired_keys = DATA_DICT.keys()
	desired_indecies = [RAW_DATA_DICT[x] for x in desired_keys]

	# Filtering out non-contiuous attributes
	unknwn_price_cnt = 0
	removed_data = 0
	for obj in data:
		if obj[RAW_DATA_DICT["price"] ] != '?':
			filtered_line = [obj[i] for i in desired_indecies]
			filtered_data = []

			# Converting viable attributes into floats
			for atrib in filtered_line:
				if atrib != '?':
					filtered_data.append( float(atrib) )
				# If attribute is unknown it is set to zero
				else:
					filtered_data.append(0)

			cleaned.append(filtered_data)
			target.append(float( obj[RAW_DATA_DICT["price"] ] ))
			removed_data += NUM_CONTIUOUS_VAR
		else:
			unknwn_price_cnt += 1

	print("Data points removed for attribute filter:", removed_data,
			"| Data points removed due to vehicles with unknown prices:", unknwn_price_cnt*TOTAL_ATRIB,
			"| Remaining datapoints =", (TOTAL_ATRIB*TOTAL_CARS) - (removed_data + unknwn_price_cnt*TOTAL_ATRIB) )
	print("Data Cleaned")
	return cleaned, target

def graphPair(f1_data, f2_data, feature1, feature2 = "price", showPlot = False):
	
	# Setting important variables for plot
	x = f1_data
	y = f2_data
	xLabel = feature1
	yLabel = feature2
	title = feature2 + " as a function of " + feature1

	# Creating the plot and placing labels
	plt.scatter(x, y)
	plt.title(title)
	plt.xlabel(xLabel)
	plt.ylabel(yLabel)

	# Setting custom values for the ticks on the x and y axis
	xstep = ( max(x) - min(x) ) / 5
	plt.xticks( np.arange(min(x), max(x)+1, xstep) )
	ystep = ( max(y) - min(y) ) / 5
	plt.yticks( np.arange(min(y), max(y)+1, ystep) )

	if showPlot:
		plt.show()
	else:
		plt.savefig(PLOTS + feature2 + "_" + feature1 + ".png", bbox_inches='tight')
		print(feature2, "v.", feature1, "| Plot Saved")
		plt.close()

	return 0

def linearT(x):
	return x

def exp_decay(x):
	return np.exp(-x)

def regression(feature, response, transformation = linearT):

	x = transformation(feature).reshape(-1, 1)
	y = response

	model = linear_model.LinearRegression().fit(x,y)

	print("Intercept (B0):", model.intercept_)
	print("Slope (B1):", model.coef_)

	r2 = model.score(x,y)
	return r2

def main():

	## Question 1A ##
	# Reading in and cleaning the data
	unfiltered_auto_data = load()
	auto_data, prices = clean_data(unfiltered_auto_data)

	# Re-organizing the data by feature
	features = DATA_DICT.keys()
	feature_data = np.transpose( np.asarray(auto_data, dtype = float) )

	"""
	## Question 1B ##
	# Plotting Price as a function of the 13 different features
	for data,feature in zip(feature_data, features):
		graphPair(data, prices, feature)

	## Question 1D ##
	# Plotting Pair-wise combinations of the 13 different features
	combos = list(itertools.combinations(feature_data, 2))
	combo_names = list(itertools.combinations(features, 2))
	for pair, names in zip(combos, combo_names):
		graphPair(pair[0], pair[1], names[0], names[1])
	"""
	
	# Performing Linear Regression using 3 different models to see which combination of the 13 features is the best predictor for price
	## Question 2B ##
	print()
	r2_len = regression(feature_data[DATA_DICT["curb-weight"]], prices)
	print("R^2 for Curb-Weight =", r2_len, '\n')
	## Question 2C ##
	r2_esize = regression(feature_data[DATA_DICT["engine-size"]], prices)
	print("R^2 for Length =", r2_esize, '\n')
	## Question 2D ##
	r2_hmpg = regression(feature_data[DATA_DICT["highway-mpg"]], prices)
	print("R^2 for Length =", r2_hmpg, '\n')

	return 0

main()
