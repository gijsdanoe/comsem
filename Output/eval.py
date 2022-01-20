import csv

baselinecount = 0
systemcount = 0
totalcount = 0

with open('output_definition.csv','r') as f:
	reader = csv.reader(f, delimiter=',')
	next(reader)
	for line in reader:
		if line[2] == '1':
			baselinecount += 1
		if line[3] == '1':
			systemcount += 1
		totalcount += 1

print('Baseline: {0}%'.format(round((baselinecount/totalcount)*100,1)))
print('System (definition): {0}%'.format(round((systemcount/totalcount)*100,1)))