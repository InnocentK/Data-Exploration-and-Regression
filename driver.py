## File: driver.py
## Date Created: 01/27/2019
## Author: Wambugu "Innocent" Kironji
## Class: ECE 580 - Introduction to Machine Learning
## Description:
##		....

DATASET = ".\\auto_data\\imports-85.data"
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

def load(filename = DATASET):
	file = open(filename, 'r')

	# Reading in all data as a list (strings for each line in the file)
	data_raw = file.readlines()

	# Seperating data into a 2D list (Rows are the different cars and columns are the attributes for the cars)
	data = data_raw.strip().split(',')

	file.close()
	return data

def clean_data(data):
	
	cleaned = []

	# Getting the indecies for all the desired car attributes
	desired_keys = DATA_DICT.keys()
	desired_indecies = [RAW_DATA_DICT[x] for x in desired_keys]

	for i in data:
		filtered_line = map(data[i].__getitem__, desired_indecies) #[data[i][j] for j in desired_indecies] 
		cleaned.append(filtered_line)

	return cleaned

def main():

	# Reading in and cleaning the data
	unfiltered_auto_data = load()
	auto_data = clean_data(unfiltered_auto_data)

	print(auto_data)

	return 0

main()