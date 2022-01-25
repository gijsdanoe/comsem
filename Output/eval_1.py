import csv

baselinecount = 0
systemcount = 0
totalcount = 0
PRON = ['i', 'we', 'you', 'he', 'she', 'it', 'they', 'this',
        'these', 'that', 'those', 'who', 'which', 'what',
        'myself', 'herself', 'himself', 'themself', 'yours']

with open('output_def_only.csv','r') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)
    for line in reader:
        if line[1].lower() not in PRON:
            if not line[1].isupper():
                if line[2] == '1':
                    baselinecount += 1
                if line[3] == '1':
                    systemcount += 1

                totalcount += 1

print(totalcount)
print(baselinecount)
print(systemcount)

print('Baseline: {0}%'.format(round((baselinecount/totalcount)*100,1)))
print('System (definition): {0}%'.format(round((systemcount/totalcount)*100,1)))
