import csv

f = open('./data/ko-en-translation.csv','r')
csv_reader = csv.reader(f)

for line in csv_reader:
  print(line)