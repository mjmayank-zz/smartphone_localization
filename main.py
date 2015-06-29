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
from scipy.stats import gaussian_kde
import collections
import random

pp = pprint.PrettyPrinter(indent=4)

def main(argv):
	cell_array = range(1, 19)
	mac_dict = {}
	times_dict = {}
	testing_data = []
	time_lens = []
	# cells = {}
	for cell_num in cell_array:
		data = []
		with open('data/cell_' + str(cell_num) + '.csv', 'rU') as csvfile:
			cell_file = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
			for row in cell_file:
				data.append(row)

	#get all the unique times
		unique_times = np.array(data)
		unique_times = unique_times[:,0]
		unique_times = sorted(set(unique_times.tolist()))
	#sort values by their time
		temp_time_dict = {}
		times = []
		for time in unique_times:
			temp_time_dict[time] = [row for row in data if time in row[0]]
			times += [len(temp_time_dict[time])]
		times_prob = calc_kde_dist(times, 150, 1)
		times_dict[cell_num] = times_prob
		graphArray(times_prob, "Number of Wi-Fi's detected in " + str(cell_num), cell_num, "num_wifis", "Number of Wi-Fi's")

	#sort out testing data
		for i in range(10):
			test_key = random.choice(temp_time_dict.keys())
			testing_data += temp_time_dict[test_key]
		for i in testing_data:
			if i in data:
				data.remove(i)

	#get all the unique mac addresses
		unique_macs = np.array(data)
		unique_macs = unique_macs[:,3]
		unique_macs = sorted(set(unique_macs.tolist()))
	#sort values by their mac addresses
		temp_mac_dict = {}
		for mac in unique_macs:
			temp_mac_dict[mac] = [row for row in data if mac in row[3]]

	#create histogram of probabilities
		for key in temp_mac_dict:
			time_lens += [len(temp_mac_dict[key])]
			rss_array = [0] * 256
			count = len(temp_mac_dict[key])
			if count > 5:
				for row in temp_mac_dict[key]:
					rss_array[abs(int(row[4]))] += 1
				if not key in mac_dict:
					mac_dict[key] = np.zeros([19, 256])
				data = np.asarray(rss_array)
				freq = []
				for i in range(len(data)):
						freq += [i] * data[i]
				mac_dict[key][cell_num] = calc_kde_dist(freq, 256, .5)

	times_dict[0] = np.asarray([0] * 150)

	print time_lens
	q75, q25 = np.percentile(time_lens, [75 ,25])
	iqr = q75 - q25
	print iqr
	print np.median(time_lens)
	# outputNumWifis(times_dict)
	# outputData(mac_dict)

	test(mac_dict, cell_array, times_dict, testing_data)

	if len(argv) > 0:
		if argv[0] == "graph":
			print "Now graphing...."
			graph(mac_dict)
		if argv[0] == "all":
			print "Now graphing...."
			graph_combined(mac_dict)

def calc_kde_dist(data, x_len, lambda_val):
	values = collections.Counter(data)
	if len(values) == 1:
		key = values.keys()
		if(key[0] + 1 < 256):
			data += [key[0] + 1]
		if(key[0] - 1 > 0):
			data += [key[0] - 1]
	density = gaussian_kde(data)
	density.covariance_factor = lambda : lambda_val
	density._compute_covariance()
	xs = np.arange(0,x_len,1)
	ys = density(xs)
	#add a small value to remove probabilities of 0
	ys = ys + np.float64(.000000001)
	ys = ys/np.float64(sum(ys))
	return ys

def test(mac_dict, cell_array, times_dict, data):
	confusion_matrix = [[0 for x in range(19)] for x in range(19)] 

#get all the unique times
	unique_times = np.array(data)
	unique_times = unique_times[:,0]
	unique_times = sorted(set(unique_times.tolist()))

#sort values by their time
	temp_time_dict = {}
	for time in unique_times:
		temp_time_dict[time] = [row for row in data if time in row[0]]

	print len(temp_time_dict.keys())
	# test_data = temp_time_dict[temp_time_dict.keys()[0]]

	for key in temp_time_dict:
		test_data = temp_time_dict[key]
		# sorted_data = test_data[np.argsort(test_data[:,4])]
		sorted_data = sorted(test_data, key=operator.itemgetter(4), reverse=True)
		true_val = int(sorted_data[0][1])
		cell = calculate_cell(mac_dict, sorted_data, times_dict)
		if cell:
			confusion_matrix[true_val][cell] += 1
	printConfMatrix(confusion_matrix)
	right = 0
	for i in range(19):
		right += confusion_matrix[i][i]
	print right, len(temp_time_dict.keys()), (right*1.0)/len(temp_time_dict.keys())

def calculate_cell(mac_dict, sorted_data, times_dict):
	probs = [1.0/17] * 19
	max_prob = 0.0
	cell = 0
	#use wifi to predict probabilities
	for i in range(10, 0, -1):
		if i >= len(sorted_data):
			for j in range(19):
				probs[j] = probs[j] * times_dict[j][len(sorted_data)]
			probs = probs/sum(probs)
			if(max(probs) > max_prob):
				cell = np.argmax(probs)
			# return cell
			i = len(sorted_data) - 1
		if sorted_data[i][3] in mac_dict:
			array_for_val = mac_dict[sorted_data[i][3]][:,int(sorted_data[i][4]) * -1]
			for i in range(len(array_for_val)):
				probs[i] = probs[i] * array_for_val[i]
			new_sum = sum(probs)
			if new_sum == 0:
				print "DIVIDING BY ZERO!!!"
			probs = [(x + .000000001) / (new_sum + (.000000001 * 19)) for x in probs]
			# for celli in range(len(probs)):
			# 	if probs[celli] > .8:
			# 		# print "i know",
			# 		return celli
			if(max(probs) > max_prob):
				cell = np.argmax(probs)
			# if(int(sorted_data[0][1]) == 4):
		# else:
		# 	print "MAC not found"
	return cell

def printConfMatrix(matrix):
	for i in range(1, 19):
		print('{:>2}').format(i),
	print
	print
	for i in matrix[1:]:
		for j in i[1:]:
			if j == 0:
				print('{:>2}').format("-"),
			else:
				print('{:>2}').format(j),
		print
	print
	for i in range(1, 19):
		print('{:>2}').format(i),
	print

def outputNumWifis(unique_times):
	with open('numWifiData.csv', 'wb') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		for cell in range(len(unique_times)):
			array = unique_times[cell].tolist()
			writer.writerow(array)

def outputData(mac_dict):
	with open('wifiData.csv', 'wb') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		for key in mac_dict:
			array = mac_dict[key]
			for i in range(len(array)):
				writer.writerow([key] + [i] + array[i].tolist())

def graphArray(data, title, cell_num, folder_name, x_label):
	plt.plot(data)
	plt.title(title)
	plt.ylabel('Probability')
	plt.xlabel(x_label)
	plt.axis([0, len(data), 0, 0.3])
	# plt.show()
	directory = 'graphs/' + folder_name + '/cell' + str(cell_num) + '/'
	if not os.path.exists(directory):
		os.makedirs(directory)
	plt.savefig(directory + title)
	plt.close()

def graph(mac_dict):
	# histogram our data with numpy
	for key in mac_dict:
		for cell in range(len(mac_dict[key])):
			data = mac_dict[key][cell]
			if sum(data) != 0:
				plt.plot(data)
				plt.title(key)
				plt.ylabel('Probability')
				plt.xlabel("RSSI Value")
				plt.axis([0, 255, 0, 0.3])
				# plt.show()
				directory = 'graphs/cell' + str(cell) + '/'
				if not os.path.exists(directory):
					os.makedirs(directory)
				plt.savefig(directory + key)
				plt.close()

def graph_combined(mac_dict):
	# histogram our data with numpy
	for key in mac_dict:
		plt.title(key)
		plt.ylabel('Probability')
		plt.xlabel("RSSI Value")
		# plt.axis([0, 255, 0, 0.3])
		min_val = 256
		max_val = 0
		cells = [2, 11]
		for cell in cells:
			data = mac_dict[key][cell]
			nonzeroind = [i for i, e in enumerate(data) if e != 0.0]
			if len(nonzeroind) > 0:
				if nonzeroind[0] < min_val:
					min_val = nonzeroind[0]
				if nonzeroind[len(nonzeroind)-1] > max_val:
					max_val = nonzeroind[len(nonzeroind)-1]
			plt.plot(data, label="cell " + str(cell))
			# plt.show()
		directory = 'graphs/different/'
		if not os.path.exists(directory):
			os.makedirs(directory)
		plt.xlim([min_val-5, max_val+5])
		lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
		plt.savefig(directory + key, bbox_extra_artists=(lgd,), bbox_inches='tight')
		plt.close()

if __name__ == "__main__":
    main(sys.argv[1:])