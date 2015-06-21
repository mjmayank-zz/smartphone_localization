import sys
import numpy as np
import csv
import pprint
import operator
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import os
from sklearn.preprocessing import normalize

pp = pprint.PrettyPrinter(indent=4)

def main(argv):
	mac_dict = {}
	# cells = {}
	data = []
	with open('queue_data.csv', 'rU') as csvfile:
		cell_file = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
		for row in cell_file:
			data.append(row)

	titles = data[0]
	data = data[1:]

	for i in range(len(data)):
		for j in range(0, len(data[0])):
			data[i][j] = float(data[i][j])

	data = np.array(data)
	data = np.delete(data, (len(data[0])-1), axis=1)

	print type(data[3][4])

	normal = []

	for column in data.T:
		# print column
		val = [(x - np.mean(column)) / np.std(column) for x in column]
		normal.append(val)

	pp.pprint(normal)
	np.savetxt("normalized.csv", np.array(normal).T, delimiter=",")

if __name__ == "__main__":
    main(sys.argv[1:])