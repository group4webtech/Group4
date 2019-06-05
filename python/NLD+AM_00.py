import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from bokeh.io import output_file, show, output_notebook
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, WheelZoomTool, ResetTool, PanTool, SaveTool, BoxSelectTool, LassoSelectTool
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.models.sources import ColumnDataSource, CDSView
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, LogTicker
from bokeh.layouts import row, column, gridplot
from bokeh.models.widgets import Tabs, Panel, MultiSelect
from bokeh.palettes import Spectral4, Spectral8, Magma6
from bokeh.plotting import figure, output_file, show
import itertools
from bs4 import BeautifulSoup
from urllib.request import urlopen

import sys, json


output_file('index.html')

separator = ";"
filename = ''

#Read data from stdin OSCAR
def read_in():
    lines = sys.stdin.readlines()
    # Since our input would only be having one line, parse our JSON data from that
    return json.loads(lines[0])

def main():
    global filename
    #get our data as an array from read_in()
    lines = read_in()

    # Sum  of all the items in the providen array
    total_sum_inArray = 0
    filename = "./upload/" + lines[0]
    print(filename)
    #return the sum to the output stream

# Start process OSCAR
if __name__ == '__main__':
    main()


# removes quotation marks at the end and start of each line (if necessary)
with open(filename, 'r') as r, open('DBL_01.csv', 'w') as w:
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
with open('DBL_01.csv', 'r') as r, open('DBL_02.csv', 'w') as w:
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
with open('DBL_02.csv', 'r') as r, open('DBL_03.csv', 'w') as w:
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
        # removes all "_" in the names of the people and replaves them by " " (spacebar)
        newline = newline.replace("_", " ")
        w.write(newline)

df = pd.read_csv('DBL_03.csv', sep=separator)     # full csv file with 1053x1053 values
# dataframe for AM
#df_new = df
max_value = df.values.max()
min_value = df.values.min()
values = df.values

counts = values.astype(float)
names = df.index.values.tolist()
names1 = df.index.values.tolist()
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

data=dict(xname=xname, yname=yname, count=counts, alphas=alpha,)

p = figure(title="Matrix Author Similarity", x_axis_location="above", x_range=list(reversed(names1)), y_range=names1)
hover_am = HoverTool(tooltips = [('Names', '@yname, @xname'), ('Value', '@count')])
p.add_tools(hover_am)
p.add_tools(BoxSelectTool())
p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "5pt"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = np.pi/3

p.rect('xname', 'yname', 0.9, 0.9, source=data, line_color=None, hover_line_color='black', alpha = 'alphas')

# SECOND AM
mapper = LinearColorMapper(palette=Magma6, low=min_value, high=max_value)
color_bar = ColorBar(color_mapper = mapper, border_line_color = None, location = (0,0))

data1=dict(xname=xname, yname=yname, count=counts)

plot = figure(title="Matrix Author Similarity", x_axis_location="above", x_range=list(reversed(names1)), y_range=names1, toolbar_location = 'below')
plot.add_tools(hover_am)
plot.add_tools(BoxSelectTool())
plot.grid.grid_line_color = None
plot.axis.axis_line_color = None
plot.axis.major_tick_line_color = None
plot.axis.major_label_text_font_size = "5pt"
plot.axis.major_label_standoff = 0
plot.xaxis.major_label_orientation = np.pi/3

plot.rect('xname', 'yname', 0.9, 0.9, source=data1, line_color=None, hover_line_color='black', fill_color={'field': 'count', 'transform': mapper})
plot.add_layout(color_bar, 'right')

alpha_panel = Panel(child = p, title = 'Alpha model')
color_panel = Panel(child = plot, title = 'Color model')

tabs1 = Tabs(tabs=[alpha_panel, color_panel])
# # END OF AM # #

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

# take 5 nodes
#df_ex = df.loc[0:4, [0,1,2,3,4]]

# Create an example graph
g=nx.DiGraph()

# add nodes and edges to the graph
weights = []

for row in df.index.values:
    for column in df.index.values:
        if  row < column:
            if (df[row][column] > 0):
                g.add_edge(row,column, weight=df[row][column])
                weights.append(df[row][column])
            else:
                 g.add_node(row)

#list_columns_names

colors = []
for n in weights:
    colors.append(10000 / n)

list_color = []
for n in range(0, len(weights)):
    list_color.append(n)

# create a dictoinary with double for loop
mapping = {old_label:new_label for old_label, new_label in itertools.zip_longest(sorted(g.nodes()), list_columns_names, fillvalue=1)}
#print(mapping)

# create a dictionary with zip()
#dictionary = dict(zip(list_columns_int, list_columns_names))
#print(dictionary)

# relabel the names of the nodes from integers back to strings
nx.relabel_nodes(g, mapping, copy=False)

plt.figure(figsize=(20,10))

pos = nx.circular_layout(g)

#nx.draw(g, pos, node_color='red', edgelist=g.edges(), edge_color=weights, arrowsize=20, with_labels=True, node_size=100, width=1.5, edge_cmap=plt.cm.Reds)

output_file('index.html')

#plt.show()

# plotting

# circular layout
plot_circle = Plot(plot_width=1000, plot_height=715,
            x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))

graph_circle = from_networkx(g, nx.circular_layout, scale=1, center=(0,0))

# !!! Specify colors with node attributes !!!
graph_circle.node_renderer.glyph = Circle(size=15,  fill_color='royalblue')
graph_circle.node_renderer.selection_glyph = Circle(size=15, fill_color='red')
graph_circle.node_renderer.hover_glyph = Circle(size=15, fill_color='green')
graph_circle.node_renderer.data_source.data['degree'] = list(zip(*g.degree))[1]
#graph_circle.node_renderer.data_source.data['colors'] = Spectral8

graph_circle.edge_renderer.glyph = MultiLine(line_color='lightskyblue', line_alpha=0.8, line_width=3)
graph_circle.edge_renderer.selection_glyph = MultiLine(line_color='red', line_width=5)
graph_circle.edge_renderer.hover_glyph = MultiLine(line_color='green', line_width=5)
graph_circle.edge_renderer.data_source.data['weight'] = weights
graph_circle.edge_renderer.glyph.line_width = {'field': 'weight'}
graph_circle.edge_renderer.data_source.data['color'] =  colors
#graph_circle.edge_renderer.glyph.line_color = {'field': 'color'}

graph_circle.selection_policy = NodesAndLinkedEdges()
graph_circle.inspection_policy = NodesAndLinkedEdges()

# !!! Hover the node attributes !!!
node_hover = HoverTool(tooltips=[('Name', '@index'), ('Degree', '@degree')])

plot_circle.add_tools(node_hover)
plot_circle.add_tools(WheelZoomTool())
plot_circle.add_tools(ResetTool())
plot_circle.add_tools(PanTool())
plot_circle.add_tools(TapTool())
plot_circle.add_tools(SaveTool())
plot_circle.add_tools(BoxSelectTool())
plot_circle.add_tools(LassoSelectTool())

plot_circle.renderers.append(graph_circle)

# spring layout
plot_spring = Plot(plot_width=1000, plot_height=715,
            x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))

graph_spring = from_networkx(g, nx.spring_layout, scale=1, center=(0,0))

# !!! Specify colors with node attributes !!!
graph_spring.node_renderer.glyph = Circle(size=15, fill_color='royalblue')
graph_spring.node_renderer.selection_glyph = Circle(size=15, fill_color='red')
graph_spring.node_renderer.hover_glyph = Circle(size=15, fill_color='green')
graph_spring.node_renderer.data_source.data['degree'] = list(zip(*g.degree))[1]
#graph_spring.node_renderer.data_source.data['colors'] = Spectral8

graph_spring.edge_renderer.glyph = MultiLine(line_color='lightskyblue', line_alpha=0.8, line_width=3)
graph_spring.edge_renderer.selection_glyph = MultiLine(line_color='red', line_width=5)
graph_spring.edge_renderer.hover_glyph = MultiLine(line_color='green', line_width=5)
graph_spring.edge_renderer.data_source.data['weight'] = weights
graph_spring.edge_renderer.glyph.line_width = {'field': 'weight'}

graph_spring.selection_policy = NodesAndLinkedEdges()
graph_spring.inspection_policy = NodesAndLinkedEdges()

# !!! Hover the node attributes !!!
plot_spring.add_tools(node_hover)
plot_spring.add_tools(WheelZoomTool())
plot_spring.add_tools(ResetTool())
plot_spring.add_tools(PanTool())
plot_spring.add_tools(TapTool())
plot_spring.add_tools(SaveTool())
plot_spring.add_tools(BoxSelectTool())
plot_spring.add_tools(LassoSelectTool())


plot_spring.renderers.append(graph_spring)

#output_notebook()
#show(plot)

# Put the legend in the upper left corner
#plot_spring.legend.location = 'top_left'

# Organize the layout

# Create two panels, one for each conference
circle_panel = Panel(child=plot_circle, title='Circle layout')
spring_panel = Panel(child=plot_spring, title='Spring layout')

# Assign the panels to Tabs
tabs = Tabs(tabs=[circle_panel, spring_panel])
p = gridplot([[tabs, tabs1]])
# Preview and save
#show(fig)
#show(fig1)

# Show the tabbed layout
show(p)

# Fetch the html file
response = urlopen('file:///C:/Users/Evgeni/AtomProjects/Bokeh%20example/index.html')
html_output = response.read()

# Parse the html file
soup = BeautifulSoup(html_output, 'html.parser')

# Format the parsed html file
strhtm = soup.prettify()

f = open('views/graphs' + '.ejs','w')

message = """{strhtm}
""".format(strhtm=strhtm)


# Print the first few characters
#print (strhtm[:225])
#print(html_output)
