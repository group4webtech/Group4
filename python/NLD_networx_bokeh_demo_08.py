import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from bokeh.io import show, output_notebook
from bokeh.models import Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, WheelZoomTool, ResetTool, PanTool, SaveTool, BoxSelectTool
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.models.sources import ColumnDataSource, CDSView
from bokeh.models.widgets import Tabs, Panel
from bokeh.palettes import Spectral4, Spectral8
from bokeh.plotting import figure, output_file, show
import itertools
from bs4 import BeautifulSoup
from urllib.request import urlopen

output_file('index.html')
#plt.show()

separator = ";"
filename = ''

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

# Preview and save
#show(fig)
#show(fig1)

# Show the tabbed layout
show(tabs)

# Fetch the html file
response = urlopen('file:///C:/Users/Evgeni/AtomProjects/Bokeh%20example/index.html')
html_output = response.read()

# Parse the html file
soup = BeautifulSoup(html_output, 'html.parser')

# Format the parsed html file
strhtm = soup.prettify()

# Print the first few characters
#print (strhtm[:225])
#print(html_output)
