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

output_file('indexmatrix.html')

filename = ''
file = ''

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
#   AM_processing_plot
###############################################################################
def AM_processing_plot(names2, source, hover_am):
    plot = figure(title="", x_axis_location="above", x_range=names2, y_range=list(reversed(names2)),toolbar_location = 'below')
    plot.plot_width = 650
    plot.plot_height = 650
    plot.grid.grid_line_color = None
    plot.axis.axis_line_color = None
    plot.axis.major_tick_line_color = None
    plot.axis.major_label_text_font_size = "5pt"
    plot.axis.major_label_standoff = 0
    plot.xaxis.major_label_orientation = np.pi/3
    plot.add_tools(hover_am)
    plot.add_tools(BoxSelectTool())

    return plot


###############################################################################
#   AM_processing
###############################################################################
def AM_processing(df_AM):
    max_value = df_AM.values.max()
    min_value = df_AM.values.min()
    values = df_AM.values
    counts = values.astype(float)
    #counts = df_AM.values
    names  = df_AM.index.values.tolist()
    names1 = df_AM.index.values.tolist()
    names.extend(names1 * (len(names1) - 1))

    def duplicate(names1, n):
        return [ele for ele in names1 for _ in range(n)]
    xname = duplicate(names1, len(names1))
    yname = names

    names2 = sorted(names1)
    names3 = sorted(names1, key = len)
    alpha = []

    for i, name in enumerate(names1):
        for j, name1 in enumerate(names1):
            alpha.append(min(counts[i, j]/4, 0.9) + 0.1)

    dataAM=dict(xname=xname, yname=yname, count=counts, alphas=alpha)
    source = ColumnDataSource(dataAM)

    #select = Select(title="Filtering:", options=['Alphabetical', 'Length'])

    hover_am = HoverTool(tooltips = [('Names', '@yname, @xname'), ('Value', '@count')])

    # color matrix
    color_palette = list(reversed(Viridis11[:8]))
    mapper = LinearColorMapper(palette=color_palette, low=min_value, high=max_value)
    color_bar = ColorBar(color_mapper = mapper, border_line_color = None, location = (0,0))
    plot_color = AM_processing_plot(names2, source, hover_am)
    plot_color.rect('xname', 'yname', 0.9, 0.9, source=source, line_color=None, hover_line_color='black', fill_color={'field': 'count', 'transform': mapper})
    plot_color.add_layout(color_bar, 'right')

    # alpha matrix
    plot_alpha = AM_processing_plot(names2, source, hover_am)
    plot_alpha.rect('xname', 'yname', 0.9, 0.9, source=source, line_color=None, hover_line_color='black', alpha = 'alphas')

    alpha_panel = Panel(child = plot_alpha, title = 'Alpha model')
    color_panel = Panel(child = plot_color, title = 'Color model')

    # Assign the AM panels to Tabs
    tabsAM_int = Tabs(tabs=[alpha_panel, color_panel])
    return tabsAM_int


###############################################################################
#   main
###############################################################################
def main():
    global filename, file
    #get our data as an array from read_in()
    lines = read_in()

    # Sum  of all the items in the providen array
    #total_sum_inArray = 0
    filename = "./upload/" + lines[0]
#    filename = "DBL.csv"
    file = lines[0]
#    print(filename)
    #return the sum to the output stream

    df_full = file_processing(filename)

    # subset dataframes
    #df_subset = df_full.loc["Jim Thomas":"James Landay", "Jim Thomas":"James Landay"]
    df_subset = df_full.loc["Jim Thomas":"Chris Buckley", "Jim Thomas":"Chris Buckley"]

    tabsAM = AM_processing(df_subset)

    # Show the tabbed layout
    show(tabsAM)

##############################################################################
#   Start process
###############################################################################
if __name__ == '__main__':
    main()


# Fetch the html file
response = urlopen('file://indexmatrix.html')
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
