import csv

def main():
	data = []
	with open('confMatrix.csv', 'rU') as csvfile:
		cell_file = csv.reader(csvfile, delimiter=',', skipinitialspace=True)
		for row in cell_file:
			data.append(row)

	confusion_matrix = [[0 for x in range(19)] for x in range(19)] 
	for row in data:
		confusion_matrix[int(row[0])][int(row[1])] += 1
	printConfMatrix(confusion_matrix)

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

if __name__ == "__main__":
    main()