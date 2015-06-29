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
	data = []
	with open('accelerometerData.txt', 'rU') as csvfile:
		cell_file = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
		for row in cell_file:
			row = [float(x) for x in row]
			data.append(row)

	calculatedValues = []

	yo = 2

	for i in range(yo, len(data)-yo):
		values = []
		prev = data[i-yo]
		curr = data[i]
		nextV = data[i+yo]
		print prev[0]
		print nextV[0]
		print
		for i in range(1, 4):
			values.append((nextV[i] - prev[i])/(nextV[0]-prev[0]))
		for i in range(1,4):
			max_v = max([prev[i], curr[i], nextV[i]])
			min_v = min([prev[i], curr[i], nextV[i]])
			diff = abs(max_v-min_v)
			values.append(diff)
		calculatedValues.append(values)

	data = np.array(calculatedValues)

	np.savetxt("queue_data.csv", data, delimiter=",")

if __name__ == "__main__":
    main(sys.argv[1:])