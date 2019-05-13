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
import itertools

# Create a graph
g=nx.Graph()

output_file('index.html')
#plt.show()

separator = ";"
filename = 'DBL'
fileext = '.csv'

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

# Create an example graph
g=nx.DiGraph()

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

# take 10 nodes
#df_ex = df.loc[0:5, [0,1,2,3,4,5]]

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

list_columns_names

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

plt.show()

# plotting

# circular layout
plot = Plot(plot_width=1000, plot_height=1000,
            x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))

graph = from_networkx(g, nx.circular_layout, scale=2, center=(0,0))

# !!! Specify colors with node attributes !!!
graph.node_renderer.glyph = Circle(size=15, fill_color='green')
graph.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=3)

# !!! Hover the node attributes !!!
node_hover = HoverTool(tooltips=[('Name', '@index'),
                                 ('Tag', '@tag'),
                                 ('Status','@status')],)
plot.add_tools(node_hover)
plot.add_tools(WheelZoomTool())
plot.add_tools(ResetTool())
plot.add_tools(PanTool())
plot.add_tools(SaveTool())
plot.add_tools(BoxSelectTool())


plot.renderers.append(graph)

# spring layout
plot_spring = Plot(plot_width=1000, plot_height=1000,
            x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))

graph_spring = from_networkx(g, nx.spring_layout, scale=2, center=(0,0))

# !!! Specify colors with node attributes !!!
graph_spring.node_renderer.glyph = Circle(size=15, fill_color='green')
graph_spring.edge_renderer.glyph = MultiLine(line_color="#CCCCCC", line_alpha=0.8, line_width=3)

# !!! Hover the node attributes !!!
node_hover = HoverTool(tooltips=[('Name', '@index'),
                                 ('Tag', '@tag'),
                                 ('Status','@status')],)
plot_spring.add_tools(node_hover)
plot_spring.add_tools(WheelZoomTool())
plot_spring.add_tools(ResetTool())
plot_spring.add_tools(PanTool())
plot_spring.add_tools(SaveTool())
plot_spring.add_tools(BoxSelectTool())

plot_spring.renderers.append(graph_spring)

#output_notebook()
#show(plot)


# Put the legend in the upper left corner
#plot_spring.legend.location = 'top_left'

# Organize the layout

# Create two panels, one for each conference
circle_panel = Panel(child=plot, title='Circle layout')
spring_panel = Panel(child=plot_spring, title='Spring layout')

# Assign the panels to Tabs
tabs = Tabs(tabs=[circle_panel, spring_panel])

# Preview and save
#show(fig)
#show(fig1)

# Show the tabbed layout
show(tabs)
