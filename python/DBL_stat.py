import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from bokeh.io import output_file, output_notebook, show
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, LogTicker, Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, WheelZoomTool, ResetTool, UndoTool, RedoTool, ZoomOutTool, ZoomInTool, PanTool, SaveTool, BoxSelectTool, LassoSelectTool
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.models.sources import ColumnDataSource, CDSView
from bokeh.layouts import row, column, gridplot
from bokeh.models.widgets import Tabs, Panel, MultiSelect, Select
from bokeh.palettes import Spectral4, Spectral8, Viridis6, Viridis11, Magma6
from bokeh.plotting import figure, output_file, show
import itertools
from math import sqrt

from bs4 import BeautifulSoup
from urllib.request import urlopen

import sys, json
import os

output_file('indexstat.html')

filename = ''
file = ''

stat_average_weigth = ''
stat_lowest_weigth = ''
stat_max_possible_edges = ''
stat_total_edges = ''
stat_sparce_percentage_of_dataset = ''
stat_total_nodes = ''
stat_average_degree = ''
stat_highest_degree = ''
stat_highest_degree_person = ''
stat_highest_weigth = ''
stat_highest_weigth_persons = ''

###############################################################################
#   Read data from stdin
###############################################################################
def read_in():
    lines = sys.stdin.readlines()
    # Since our input would only be having one line, parse our JSON data from that
    return json.loads(lines[0])


###############################################################################
#   file_processing
###############################################################################
def file_processing(filename):
    separator = ";"

    new_filename = filename.replace(".csv", "_DBL.csv")
    if new_filename == filename:
        new_filename += "_DBL.csv"

    with open(filename, 'r') as r, open(new_filename, 'w') as w:
        for num, line in enumerate(r):
            if num >= 0:

                # removes quotation marks at the end and start of each line (if necessary)
                if line[0] == "\"" and line[-1] == "\"":
                    newline = line[1:-1]
                elif line[0] == "\"" and line[-1] == "\n" and line[-2] == "\"":
                    newline = line[1:-2] + "\n"
                else:
                    newline = line

                # removes the first coma at the beginning of each line (if necessary)
                line = newline
                if line[0] == separator:
                    newline = line[1:]
                else:
                    newline = line

                # removes comas at the end of each line (if necessary)
                line = newline
                if line[-1] == separator:
                    newline = line[:-1]
                elif line[-1] == "\n" and line[-2] == separator:
                    newline = line[:-2] + "\n"
                else:
                    newline = line
            else:
                newline = line
            # removes all "_" in the names of the people and replaves them by " " (spacebar)
            newline = newline.replace("_", " ")
            w.write(newline)

    # full dataframe
    df_full = pd.read_csv(new_filename, sep=separator)     # full csv file with 1053x1053 values
    return df_full


###############################################################################
#   main
###############################################################################
def main():
    global filename, file

    global stat_average_weigth
    global stat_lowest_weigth
    global stat_max_possible_edges
    global stat_total_edges
    global stat_sparce_percentage_of_dataset
    global stat_total_nodes
    global stat_average_degree
    global stat_highest_degree
    global stat_highest_degree_person
    global stat_highest_weigth
    global stat_highest_weigth_persons

    #get our data as an array from read_in()
#    lines = read_in()

    # Sum  of all the items in the providen array
    #total_sum_inArray = 0
#    filename = "./upload/" + lines[0]
    filename = "DBL.csv"
#    file = lines[0]
#    print(filename)
    #return the sum to the output stream

    df_full = file_processing(filename)

    # subset dataframes
    df = df_full

    # get original column names from df
    list_columns_names = df.columns

    # change the column labels from strings to integers
    list_columns_int = []

    for number in range(0, len(df.index.values)):
        list_columns_int.append(number)

    df.columns = list_columns_int

    # change the rows labels/ indexes from strings to integers
    df['index'] = list_columns_int
    df.set_index("index", inplace=True)

    # Create a graph with 1-way edges for faster painting
    g=nx.DiGraph()
    for row in df.index.values:
        g.add_node(row)
        for column in df.index.values:
            if  row < column:
                if (df[row][column] > 0):
                    g.add_edge(row, column, weight=df[row][column])

    weights = []
    # Create a separate graph with 2-way edges only to calculate weights
    g_w=nx.DiGraph()
    for row in df.index.values:
        g_w.add_node(row)
        for column in df.index.values:
            if  row != column:
                if (df[row][column] > 0):
                    g_w.add_edge(row, column, weight=df[row][column])
                    weights.append(df[row][column])


    # average weigth of an edge
    stat_average_weigth = ((int)(sum(weights) * 10000 / len(weights))) / 100
    print("average weigth of an edge: ",  stat_average_weigth)

    # minimum weigth of an edge
    stat_lowest_weigth = min(weights)
    print("minimum weigth of an edge: ", stat_lowest_weigth)

    # maximum possible number of edges
    stat_max_possible_edges = len(g.nodes) * len(g.nodes)
    print("maximum possible number of edges: ", stat_max_possible_edges)

    # total number of edges
    stat_total_edges = len(g.edges)
    print("total number of edges: ", stat_total_edges)

    # sparce percentage of dataset
    stat_sparce_percentage_of_dataset = ((int)(stat_total_edges * 10000 / stat_max_possible_edges)) / 100
    print("sparce percentage of dataset: ", stat_sparce_percentage_of_dataset)

    # total number of nodes
    stat_total_nodes = len(g.nodes)
    print("total number of nodes: ", stat_total_nodes)

    # average node degree
    stat_average_degree = (int)(sum(list(list(zip(*g.degree))[1])) / len(g.nodes))
    print("average node degree: ", stat_average_degree)

    # lowest node degree
    stat_lowest_degree = min(list(list(zip(*g.degree))[1]))
    print("lowest node degree: ", stat_lowest_degree)

    # create a dictoinary with double for loop
    mapping = {old_label:new_label for old_label, new_label in itertools.zip_longest(sorted(g.nodes()), list_columns_names, fillvalue=1)}

    # relabel the names of the nodes from integers back to strings
    nx.relabel_nodes(g, mapping, copy=False)


    # highest node degree - for person' 'X'
    stat_highest_degree = max(list(list(zip(*g.degree))[1]))
    print("highest node degree: ", stat_highest_degree)

    i = 0
    curent_max_degree = list(list(zip(*g.degree))[1])[i]
    curent_max_degree_index = i
    for node in list(list(zip(*g.degree))[0]):
        if (list(list(zip(*g.degree))[1])[i] > curent_max_degree):
            curent_max_degree = list(list(zip(*g.degree))[1])[i]
            curent_max_degree_index = i
        i += 1
    stat_highest_degree = curent_max_degree
    print("highest node degree2: ", stat_highest_degree)
    stat_highest_degree_person = list(list(zip(*g.degree))[0])[curent_max_degree_index]
    print(stat_highest_degree_person)


    # maximum weigth of an edge - between persons 'u' and 'v'
    stat_highest_weigth = max(weights)
    print("maximum weigth of an edge: ", stat_highest_weigth)

    for u,v,d in g.edges(data=True):
        if (d['weight'] == stat_highest_weigth):
            stat_highest_weigth_persons = u + ", " + v
            print(stat_highest_weigth_persons)


##############################################################################
#   Start process
###############################################################################
if __name__ == '__main__':
    main()


# Fetch the html file
response = urlopen('file://indexstat.html')
html_output = response.read()

# Parse the html file
soup = BeautifulSoup(html_output, 'html.parser')
ww = "qqqq"

# Format the parsed html file
strhtm = soup.prettify()

fileWithNo = os.path.splitext(file)[0]

f = open('views/graphs/' + fileWithNo + 'nodelink.ejs','w')

message = """{strhtm}
""".format(strhtm=strhtm)

#print(message)

f.write(message)
f.close()
