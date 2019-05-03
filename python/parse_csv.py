import csv

with open('upload/GephiMatrix_co-authorship.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)

    for line in csv_reader:
        print(line)
