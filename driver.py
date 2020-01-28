## File: driver.py
## Date Created: 01/27/2019
## Author: Wambugu "Innocent" Kironji
## Class: ECE 580 - Introduction to Machine Learning
## Description:
##		....

import matplotlib.pyplot as plt
import numpy as np

WINDOWS = "\\"
UNIX = "/"
OS = UNIX
DATASET = "." + OS + "auto_data" + OS + "imports-85.data"
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

def graphPrice2Feature(feature_data, prices, feature):
	
	# Setting important variables for plot
	y = feature_data
	x = prices
	yLabel = feature
	xLabel = "Price"
	title = "Price as a function of " + feature

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

	plt.show()

def main():

	# Reading in and cleaning the data
	unfiltered_auto_data = load()
	auto_data, prices = clean_data(unfiltered_auto_data)

	# Re-organizing the data by feature
	features = DATA_DICT.keys()
	feature_data = np.transpose( np.asarray(auto_data, dtype = float) )

	# Plotting Price as a function of the 13 different features
	for data,feature in zip(feature_data, features):
		graphPrice2Feature(data, prices, feature)

	return 0

main()
