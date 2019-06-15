import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from bokeh.io import show, output_notebook
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, WheelZoomTool, ResetTool, PanTool, SaveTool, BoxSelectTool
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.models.sources import ColumnDataSource, CDSView
from bokeh.models.widgets import Tabs, Panel
from bokeh.palettes import Spectral4
from bokeh.plotting import figure, output_file, show
from math import sqrt
import itertools
import os

# Create a graph
g=nx.Graph()

output_file('index.html')
#plt.show()

separator = ";"
filename = 'DBL'
fileext = '.csv'

# Code specifying directory of csv
os.chdir(r"C:\Users\20181214\Desktop\Q4\DBL_Webtech")

# removes quotation marks at the end and start of each line (if necessary)
with open(filename + fileext, 'r') as r, open(filename + '_01' + fileext, 'w') as w:
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
with open(filename + '_01' + fileext, 'r') as r, open(filename + '_02' + fileext, 'w') as w:
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
with open(filename + '_02' + fileext, 'r') as r, open(filename + '_03' + fileext, 'w') as w:
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

df = pd.read_csv(filename + '_03' + fileext, sep=separator)     # full csv file with 1053x1053 values

# change the column labels from strings to integers

list_columns = []

for number in range(0, len(df.index.values)):
    list_columns.append(number)

df.columns = list_columns


# change the rows labels/ indexes from strings to integers
df['index'] = list_columns
df.set_index("index", inplace=True)


#df_ex = df.loc[0:24, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]
df_ex = df

weights = []

for row in df_ex.index.values:
    for column in df_ex.index.values:
        if  (row < column) and (df_ex[row][column] > 0):
            g.add_edge(row,column, weight=df_ex[row][column])
            weights.append(df_ex[row][column])


plt.figure(figsize=(15,15))

pos = nx.spring_layout(g)
#nx.draw(g, with_labels=False, pos, node_color='b', node_size=20, arrowstyle='->', arrowsize=20, font_size=10, font_weight="bold", edgelist=g.edges(), edge_color=weights, width=10.0, edge_cmap=plt.cm.Blues)
#nx.draw(g, pos, node_color='red', edgelist=g.edges(), edge_color=weights, arrowsize=20, with_labels=True, nodesize=10, width=2.0, edge_cmap=plt.cm.Reds)

#plt.savefig('NLD.png')
#plt.show()

pos=nx.fruchterman_reingold_layout(g)


r = 1.5*(max(g.degree())[1]+0.5)/sqrt(g.number_of_nodes())

ly = nx.fruchterman_reingold_layout(g,k=r, iterations=50,pos=nx.fruchterman_reingold_layout(g))

d = dict(g.degree)

nx.draw(g, ly, nodelist=d.keys(),  node_color='blue', edgelist=g.edges(), edge_color=weights, arrowsize=20, with_labels=False, node_size=[v * 10 for v in d.values()], width=2.0, edge_cmap=plt.cm.Reds)
plt.savefig("Graph.png", format="PNG")
plt.show()
