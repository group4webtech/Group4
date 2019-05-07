import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

separator = ";"
csv_filename = 'DBL.csv'

# 'GephiMatrix_author_similarity.csv', 'GephiMatrix_co-authorship.csv', 'GephiMatrix_co-citation.csv'

g=nx.Graph()

# removes quotation marks at the end and start of each line (if necessary)
with open(csv_filename, 'r') as r, open('NLD_01.csv', 'w') as w:
    for num, line in enumerate(r):
        if num >= 0:
            if line[0] == "\"" and line[-1] == "\"":
                newline = line[1:-1]
            elif line[0] == "\"" and line[-1] == "\n" and line[-2] == "\"":
                newline = line[1:-2] + "\n"
            else:
                newline = line
        else:
            newline = line
        w.write(newline)

# removes the first coma at the beginning of each line (if necessary)
with open('NLD_01.csv', 'r') as r, open('NLD_02.csv', 'w') as w:
    for num, line in enumerate(r):
        if num >= 0:
            if line[0] == separator:
                newline = line[1:]
            else:
                newline = line
        else:
            newline = line
        w.write(newline)

# removes comas at the end of each line (if necessary)
with open('NLD_02.csv', 'r') as r, open('NLD_03.csv', 'w') as w:
    for num, line in enumerate(r):
        if num >= 0:
            if line[-1] == separator:
                newline = line[:-1]
            elif line[-1] == "\n" and line[-2] == separator:
                newline = line[:-2] + "\n"
            else:
                newline = line
        else:
            newline = line
        w.write(newline)

df = pd.read_csv('NLD_03.csv', sep=separator)     # full csv file with 1053x1053 values

# change the column labels from strings to integers

list_columns = []

for number in range(0, len(df.index.values)):
    list_columns.append(number)

df.columns = list_columns


# change the rows labels/ indexes from strings to integers
df['index'] = list_columns
df.set_index("index", inplace=True)


df_ex = df.loc[0:24, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]


weights = []

for row in df_ex.index.values:
    for column in df_ex.index.values:
        if  (row < column) and (df_ex[row][column] > 0):
            g.add_edge(row,column, weight=df_ex[row][column])
            weights.append(df_ex[row][column])


plt.figure(figsize=(20,10))

pos = nx.spring_layout(g)
#nx.draw(g, with_labels=False, pos, node_color='b', node_size=20, arrowstyle='->', arrowsize=20, font_size=10, font_weight="bold", edgelist=g.edges(), edge_color=weights, width=10.0, edge_cmap=plt.cm.Blues)
nx.draw(g, pos, node_color='red', edgelist=g.edges(), edge_color=weights, arrowsize=20, with_labels=True, nodesize=10, width=2.0, edge_cmap=plt.cm.Reds)

plt.savefig('NLD.png')
plt.show()
