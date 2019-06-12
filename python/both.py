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

output_file('indexboth.html')

separator = ";"
filename = ''
file = ''

#Read data from stdin OSCAR
def read_in():
    lines = sys.stdin.readlines()
    # Since our input would only be having one line, parse our JSON data from that
    return json.loads(lines[0])

def main():
    global filename, file
    #get our data as an array from read_in()
#    lines = read_in()

    # Sum  of all the items in the providen array
    #total_sum_inArray = 0
#    filename = "./upload/" + lines[0]
    filename = "DBL.csv"
#    file = lines[0]
#    print(filename)
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

# subset
df_subset = df.loc["Jim Thomas":"Chris Buckley", "Jim Thomas":"Chris Buckley"]
df_new = df.loc["Jim Thomas":"Chris Buckley", "Jim Thomas":"Chris Buckley"]
df = df_subset

max_value = df_new.values.max()
min_value = df_new.values.min()
values = df_new.values
counts = values.astype(float)
names = df_new.index.values.tolist()
names1 = df_new.index.values.tolist()
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

plot_alpha = figure(title="", x_axis_location="above", x_range=names2, y_range=list(reversed(names2)),toolbar_location = 'below')

plot_alpha.add_tools(hover_am)
plot_alpha.add_tools(BoxSelectTool())
plot_alpha.plot_width = 650
plot_alpha.plot_height = 650
plot_alpha.grid.grid_line_color = None
plot_alpha.axis.axis_line_color = None
plot_alpha.axis.major_tick_line_color = None
plot_alpha.axis.major_label_text_font_size = "5pt"
plot_alpha.axis.major_label_standoff = 0
plot_alpha.xaxis.major_label_orientation = np.pi/3

plot_alpha.rect('xname', 'yname', 0.9, 0.9, source=source, line_color=None, hover_line_color='black', alpha = 'alphas')

# SECOND AM
color_palette = list(reversed(Viridis11[:8]))
mapper = LinearColorMapper(palette=color_palette, low=min_value, high=max_value)
color_bar = ColorBar(color_mapper = mapper, border_line_color = None, location = (0,0))

plot_color = figure(title="", x_axis_location="above", x_range=names2, y_range=list(reversed(names2)),toolbar_location = 'below')
plot_color.add_tools(hover_am)
plot_color.add_tools(BoxSelectTool())
plot_color.plot_width = 650
plot_color.plot_height = 650
plot_color.grid.grid_line_color = None
plot_color.axis.axis_line_color = None
plot_color.axis.major_tick_line_color = None
plot_color.axis.major_label_text_font_size = "5pt"
plot_color.axis.major_label_standoff = 0
plot_color.xaxis.major_label_orientation = np.pi/3

plot_color.rect('xname', 'yname', 0.9, 0.9, source=source, line_color=None, hover_line_color='black', fill_color={'field': 'count', 'transform': mapper})
plot_color.add_layout(color_bar, 'right')

alpha_panel = Panel(child = plot_alpha, title = 'Alpha model')
color_panel = Panel(child = plot_color, title = 'Color model')

tabsAM = Tabs(tabs=[alpha_panel, color_panel])
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

# Create an example graph
g=nx.DiGraph()


# Making a function to map color to edges
color_palette = list(reversed(Viridis11[:8]))
w_max = df.values.max()
w_min = df.values.min()
step = (w_max-w_min)/(len(color_palette)-1)

# add nodes and edges to the graph
weights = []
colors = []
for row in df.index.values:
    g.add_node(row)
    for column in df.index.values:
        if  row != column:
            if (df[row][column] > 0):
                color_index = int((df[row][column] - w_min) / step)
                g.add_edge(row, column, weight=df[row][column], color=color_palette[color_index])
                weights.append(df[row][column])
                colors.append(color_palette[color_index])

# loop over all nodes to find neighbors and set min, max, sum for egdes weights connected to a node
node_attr_dict = {}
for n in list_columns_int:
    node_weight_list = []
    for nb in nx.neighbors(g, n):
        node_weight_list.append(nx.get_edge_attributes(g,'weight')[n, nb])
    if len(node_weight_list) != 0:
        node_min_weight = min(node_weight_list)
        node_max_weight = max(node_weight_list)
        node_sum_weight = sum(node_weight_list)
    else:
        node_min_weight = 0
        node_max_weight = 0
        node_sum_weight = 0
    node_attr_dict.update({n:{'minweight':node_min_weight, 'maxweight':node_max_weight, 'sumweight':node_sum_weight}})
nx.set_node_attributes(g, node_attr_dict)

# create a dictoinary with double for loop
mapping = {old_label:new_label for old_label, new_label in itertools.zip_longest(sorted(g.nodes()), list_columns_names, fillvalue=1)}

# relabel the names of the nodes from integers back to strings
nx.relabel_nodes(g, mapping, copy=False)

# Organize common layouts' size for NLD
NLD_width  = 730
NLD_height = 690

color_mapper = LinearColorMapper(palette=color_palette, low=w_min, high=w_max)
color_bar = ColorBar(color_mapper = color_mapper, border_line_color = None, location = (0,0))




# circular layout
plot_circle = Plot(plot_width=NLD_width, plot_height=NLD_height,
            x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))

graph_circle = from_networkx(g, nx.circular_layout, scale=1, center=(0,0))

list_degree = []
for degree in list(zip(*g.degree))[1]:
    list_degree.append(int(degree/2))

# nodes and egdes attributes
graph_circle.node_renderer.data_source.data['degree'] = list_degree
graph_circle.edge_renderer.data_source.data['weight'] = weights
graph_circle.edge_renderer.data_source.add(colors, 'color')

graph_circle.node_renderer.glyph            = Circle(size=10, fill_alpha=0.8, fill_color='royalblue')
graph_circle.node_renderer.selection_glyph  = Circle(size=10, fill_alpha=0.8, fill_color='red')
graph_circle.node_renderer.hover_glyph      = Circle(size=10, fill_alpha=0.8, fill_color='yellow')

graph_circle.edge_renderer.glyph            = MultiLine(line_width=3, line_alpha=0.8, line_color='color')
graph_circle.edge_renderer.selection_glyph  = MultiLine(line_width=4, line_alpha=0.8, line_color='red')
graph_circle.edge_renderer.hover_glyph      = MultiLine(line_width=4, line_alpha=0.8, line_color='yellow')
graph_circle.edge_renderer.glyph.line_width = {'field': 'weight'}

graph_circle.selection_policy = NodesAndLinkedEdges()
graph_circle.inspection_policy = NodesAndLinkedEdges()

# !!! Hover the node attributes !!!
plot_circle.add_tools(WheelZoomTool())
plot_circle.add_tools(ZoomOutTool())
plot_circle.add_tools(ZoomInTool())
plot_circle.add_tools(ResetTool())
plot_circle.add_tools(UndoTool())
plot_circle.add_tools(RedoTool())
plot_circle.add_tools(PanTool())
plot_circle.add_tools(TapTool())
plot_circle.add_tools(SaveTool())
plot_circle.add_tools(BoxSelectTool())
plot_circle.add_tools(LassoSelectTool())

node_hover = HoverTool(tooltips=[('Name', '@index'), ('Degree', '@degree'),
                                    ('Min Weight', '@minweight'), ('Max Weight', '@maxweight'), ('Sum Weight', '@sumweight')])
plot_circle.add_tools(node_hover)

plot_circle.add_layout(color_bar, 'right')

plot_circle.renderers.append(graph_circle)



# spring layout
plot_spring = Plot(plot_width=NLD_width, plot_height=NLD_height,
            x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))

graph_spring = from_networkx(g, nx.spring_layout, scale=1, center=(0,0))

list_degree = []
for degree in list(zip(*g.degree))[1]:
    list_degree.append(int(degree/2))

# nodes and egdes attributes
graph_spring.node_renderer.data_source.data['degree'] = list_degree
graph_spring.edge_renderer.data_source.data['weight'] = weights
graph_spring.edge_renderer.data_source.add(colors, 'color')

graph_spring.node_renderer.glyph            = Circle(size=10, fill_alpha=0.8, fill_color='royalblue')
graph_spring.node_renderer.selection_glyph  = Circle(size=10, fill_alpha=0.8, fill_color='red')
graph_spring.node_renderer.hover_glyph      = Circle(size=10, fill_alpha=0.8, fill_color='yellow')

graph_spring.edge_renderer.glyph            = MultiLine(line_width=3, line_alpha=0.8, line_color='color')
graph_spring.edge_renderer.selection_glyph  = MultiLine(line_width=4, line_alpha=0.8, line_color='red')
graph_spring.edge_renderer.hover_glyph      = MultiLine(line_width=4, line_alpha=0.8, line_color='yellow')
graph_spring.edge_renderer.glyph.line_width = {'field': 'weight'}

graph_spring.selection_policy = NodesAndLinkedEdges()
graph_spring.inspection_policy = NodesAndLinkedEdges()

# !!! Hover the node attributes !!!
plot_spring.add_tools(WheelZoomTool())
plot_spring.add_tools(ZoomOutTool())
plot_spring.add_tools(ZoomInTool())
plot_spring.add_tools(ResetTool())
plot_spring.add_tools(UndoTool())
plot_spring.add_tools(RedoTool())
plot_spring.add_tools(PanTool())
plot_spring.add_tools(TapTool())
plot_spring.add_tools(SaveTool())
plot_spring.add_tools(BoxSelectTool())
plot_spring.add_tools(LassoSelectTool())

plot_spring.add_tools(node_hover)

plot_spring.add_layout(color_bar, 'right')

plot_spring.renderers.append(graph_spring)



# force-directed representation
plot_fd = Plot(plot_width=NLD_width, plot_height=NLD_height,
            x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))

r = 5/sqrt(g.number_of_nodes()) # changed this to make it work

my_points=nx.fruchterman_reingold_layout(g)
graph_fd = from_networkx(g, nx.fruchterman_reingold_layout(g,k=r, iterations=100, pos=my_points, scale=1, center=(0,0)))

#posxy=nx.fruchterman_reingold_layout(g)
#for i in range(len(posxy()):
#    posxy
#radii = np.random.random(size=N) * 1.5
my_colors = []
for key, value in my_points.items():
    x = value[0] + 1.0 # x = -1 .. +1, so move to 0 ... 2
    y = value[1] + 1.0 # y = -1 .. +1, so move to 0 ... 2
    my_colors.append( "#%02x%02x%02x" % (int(50+100*x), int(30+100*y), 150) )

list_degree = []
for degree in list(zip(*g.degree))[1]:
    list_degree.append(int(degree/2))

# nodes and egdes attributes
graph_fd.node_renderer.data_source.data['degree'] = list_degree
graph_fd.node_renderer.data_source.data['degree2'] = [(x+2)*1050 for x in graph_fd.node_renderer.data_source.data['degree']]
graph_fd.node_renderer.data_source.data['nodesize'] = [x/(g.number_of_nodes()+150) for x in graph_fd.node_renderer.data_source.data['degree2']]

graph_fd.node_renderer.data_source.data['my_fill_color'] = my_colors
graph_fd.edge_renderer.data_source.data['weight'] = weights
graph_fd.edge_renderer.data_source.add(colors, 'color')

graph_fd.node_renderer.glyph            = Circle(size ="nodesize",  fill_color='my_fill_color', fill_alpha=0.85)
graph_fd.node_renderer.selection_glyph  = Circle(size=15, fill_alpha=0.8, fill_color='red')
graph_fd.node_renderer.hover_glyph      = Circle(size=15, fill_alpha=0.8, fill_color='yellow')

graph_fd.edge_renderer.glyph            = MultiLine(line_width=3, line_alpha=0.8, line_color='color')
graph_fd.edge_renderer.selection_glyph  = MultiLine(line_width=4, line_alpha=0.8, line_color='red')
graph_fd.edge_renderer.hover_glyph      = MultiLine(line_width=4, line_alpha=0.8, line_color='yellow')
graph_fd.edge_renderer.glyph.line_width = {'field': 'weight'}

graph_fd.selection_policy = NodesAndLinkedEdges()
graph_fd.inspection_policy = NodesAndLinkedEdges()

# !!! Hover the node attributes !!!
plot_fd.add_tools(WheelZoomTool())
plot_fd.add_tools(ZoomOutTool())
plot_fd.add_tools(ZoomInTool())
plot_fd.add_tools(ResetTool())
plot_fd.add_tools(UndoTool())
plot_fd.add_tools(RedoTool())
plot_fd.add_tools(PanTool())
plot_fd.add_tools(TapTool())
plot_fd.add_tools(SaveTool())
plot_fd.add_tools(BoxSelectTool())
plot_fd.add_tools(LassoSelectTool())

plot_fd.add_tools(node_hover)

plot_fd.add_layout(color_bar, 'right')

plot_fd.renderers.append(graph_fd)



# Organize the layout

# Create two panels, one for each conference
circle_panel = Panel(child=plot_circle, title='Circle layout')
spring_panel = Panel(child=plot_spring, title='Spring layout')
fd_panel     = Panel(child=plot_fd, title='Force-Directed layout')

# Assign the panels to Tabs
tabsNLD = Tabs(tabs=[circle_panel, spring_panel, fd_panel])
tabsAM = Tabs(tabs=[alpha_panel, color_panel])
grid = gridplot([[tabsNLD, tabsAM]])

# Show the tabbed layout
show(grid)

# Fetch the html file
#response = urlopen('file:///C:/Users/Oscar/Documents/tue/q4/Webtech/v2/indexnodelink.html')
#html_output = response.read()

# Parse the html file
#soup = BeautifulSoup(html_output, 'html.parser')
#ww = "qqqq"

# Format the parsed html file
#strhtm = soup.prettify()

#fileWithNo = os.path.splitext(file)[0]

#f = open('views/graphs/' + fileWithNo + 'nodelink.ejs','w')

#message = """{strhtm}
#""".format(strhtm=strhtm)

#print(message)

#f.write(message)
#f.close()
