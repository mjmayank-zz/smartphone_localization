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
# from sklearn.preprocessing import normalize

pp = pprint.PrettyPrinter(indent=4)

def main(argv):
	raw_to_calc()
	normalizer()

def normalizer():
	data = []
	with open('queue_data.csv', 'rU') as csvfile:
		cell_file = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
		for row in cell_file:
			row = [float(x) for x in row]
			data.append(row)

	data = np.array(data)

	normal = []
	means = []
	stds = []

	for i in range(len(data.T)-1):
		# print column
		column = data.T[i]
		means.append(np.mean(column))
		stds.append(np.std(column))
		val = [(x - np.mean(column)) / np.std(column) for x in column]
		val = [abs(x) for x in val]
		normal.append(val)
	normal.append(data.T[len(data.T)-1])
	print data.T[len(data.T)-1]

	# pp.pprint(normal)
	# print means
	# print stds
	data = np.array(normal).T
	print len(data[0])
	print len(stds)
	print stds + [0.0]
	print means + [0.0]
	data = np.insert(data, 0, stds + [0.0], axis=0)
	data = np.insert(data, 0, means + [0.0], axis=0)

	np.savetxt("normalized.csv", data, delimiter=",")

def raw_to_calc():
	data = []
	with open('accelerometerData.txt', 'rU') as csvfile:
		cell_file = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
		for row in cell_file:
			row = [float(x) for x in row]
			data.append(row)

	calculatedValues = []

	yo = 9

	for i in range(yo, len(data)-yo, 1):
		print i
		values = []
		time = []
		x_arr = []
		y_arr = []
		z_arr = []
		# print yo/2
		for j in range(-(yo/2), yo/2+1):
			time += [data[i+j][0]]
			x_arr += [data[i+j][1]]
			y_arr += [data[i+j][2]]
			z_arr += [data[i+j][3]]
		values.append((x_arr[yo-1]-x_arr[0])/(time[yo-1]-time[0]))
		values.append((y_arr[yo-1]-y_arr[0])/(time[yo-1]-time[0]))
		values.append((z_arr[yo-1]-z_arr[0])/(time[yo-1]-time[0]))
		values.append(np.var(x_arr))
		values.append(np.var(y_arr))
		values.append(np.var(z_arr))
		# print data[i][len(data[i])-1]
		for j in range(1,4):
			diff = abs(max(x_arr)-min(x_arr))
			values.append(diff)
		values.append(data[i][4])
		# print data[i]
		calculatedValues.append(values)

	data = np.array(calculatedValues)
	np.savetxt("queue_data.csv", data, delimiter=",")

if __name__ == "__main__":
    main(sys.argv[1:])