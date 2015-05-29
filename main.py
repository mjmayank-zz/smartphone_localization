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

pp = pprint.PrettyPrinter(indent=4)

def main(argv):
	cell_array = [2, 3, 4, 11]
	mac_dict = {}
	# cells = {}
	for cell_num in cell_array:
		data = []
		with open('cell_' + str(cell_num) + ".csv", 'rU') as csvfile:
			cell_file = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
			for row in cell_file:
				data.append(row)

	#get all the unique mac addresses
		unique_macs = np.array(data)
		unique_macs = unique_macs[:,3]
		unique_macs = sorted(set(unique_macs.tolist()))
		# print all_macs

	#sort values by their mac address
		temp_mac_dict = {}
		for mac in unique_macs:
			temp_mac_dict[mac] = [row for row in data if mac in row[3]]

	#create histogram of probabilities
		for key in temp_mac_dict:
			rss_array = [0.0] * 256
			count = 0.0
			for row in temp_mac_dict[key]:
				rss_array[abs(int(row[4]))] += 1
				count += 1
			if count > 3:
				if not key in mac_dict:
					mac_dict[key] = np.zeros([19, 256])
				mac_dict[key][cell_num] = np.asarray([x / count for x in rss_array])

	test(mac_dict, cell_array)
	if len(argv) > 0:
		if argv[0] == "graph":
			print "Now graphing...."
			graph(mac_dict)
		if argv[0] == "all":
			print "Now graphing...."
			graph_combined(mac_dict)

def test(mac_dict, cell_array):
	data = []
	with open("testing.csv", 'rU') as csvfile:
		cell_file = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
		for row in cell_file:
			data.append(row)

#get all the unique mac addresses
	unique_times = np.array(data)
	unique_times = unique_times[:,0]
	unique_times = sorted(set(unique_times.tolist()))
	# print all_macs

#sort values by their mac address
	temp_time_dict = {}
	for time in unique_times:
		temp_time_dict[time] = [row for row in data if time in row[0]]

	test_data = temp_time_dict[temp_time_dict.keys()[0]]


	# test_data = {"5c:96:9d:65:76:8d":	-71,
	# 			 "80:ea:96:eb:1e:fc":	-65,
	# 			 "60:36:dd:cb:c1:4f":	-55,
	# 			 "1c:aa:07:7b:39:13":	-71,
	# 			 "1c:aa:07:b0:7a:b3":	-74,
	# 			 "00:22:f7:21:d0:38":	-94,
	# 			 "1c:aa:07:b0:7a:b0":	-73,
	# 			 "1c:aa:07:b0:7a:b2":	-73,
	# 			 "1c:aa:07:7b:39:10":	-73,
	# 			 "1c:aa:07:b0:7a:b1":	-71,
	# 			 "1c:aa:07:6e:31:a3":	-75,
	# 			 "1c:aa:07:6e:31:a2":	-75,
	# 			 "00:0c:f6:a6:3c:e8":	-76,
	# 			 "1c:aa:07:7b:37:03":	-80,
	# 			 "48:f8:b3:40:f1:a9":	-96,
	# 			 "8c:21:0a:9a:98:f8":	-96,
	# 			 "1c:aa:07:6e:31:a1":	-75}

	# sorted_data = test_data[np.argsort(test_data[:,4])]
	sorted_data = sorted(test_data, key=operator.itemgetter(4), reverse=True)
	pp.pprint(sorted_data)
	# pp.pprint(sorted_data)
	# print sorted_data[0][0]
	# pp.pprint(mac_dict[sorted_data[0][0]])
	probs = [1.0/4] * 19
	print 
	print
	for i in range(6):
		if sorted_data[i][3] in mac_dict:
			# print "-"
			array_for_val = mac_dict[sorted_data[i][3]][:,int(sorted_data[i][4]) * -1]
			for i in range(len(array_for_val)):
				probs[i] *= array_for_val[i]
			# pp.pprint(array_for_val)
			# print "Sum: ", sum(probs)
			# print "range: ", cell_range
			new_sum = sum(probs)
			probs = [x / new_sum for x in probs]
			print "Prob: ", probs
		else:
			print "MAC not found"

def graph(mac_dict):
	# histogram our data with numpy
	for key in mac_dict:
		for cell in range(len(mac_dict[key])):
			data = mac_dict[key][cell]
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