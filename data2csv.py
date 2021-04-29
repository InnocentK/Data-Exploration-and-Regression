## File: data2csv.py
## Date Created: 02/04/2019
## Author: Wambugu "Innocent" Kironji
## Class: ECE 580 - Introduction to Machine Learning
## Description:
##		Converting a .data file that is comma delimited to a .csv

WINDOWS = "\\"
UNIX = "/"
OS = UNIX
DATASET = "." + OS + "auto_data" + OS + "imports-85.data"

# Loads the data from .data file into a list then outputs to .csv
def loadNSave(filename = DATASET, out_fname = "imports-85.csv"):

	file = open(filename, 'r')
	out = open(out_fname, 'w')

	# Reading in all data as a list (strings for each line in the file)
	data_raw = file.readlines()

	# Writing out the list of strings line by line
	out.writelines(data_raw)

	file.close()
	out.close()
	print("Data Written")
	return 0

def main():
	loadNSave()
	return 0

main()