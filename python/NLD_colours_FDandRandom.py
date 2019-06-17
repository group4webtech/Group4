import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from bokeh.io import output_file, output_notebook, show
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, LogTicker, Plot, Range1d, MultiLine, Circle, HoverTool, TapTool, BoxSelectTool, WheelZoomTool, BoxZoomTool, ResetTool, UndoTool, RedoTool, ZoomOutTool, ZoomInTool, PanTool, SaveTool, BoxSelectTool, LassoSelectTool
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.models.sources import ColumnDataSource, CDSView
from bokeh.layouts import row, column, gridplot
from bokeh.models.widgets import Tabs, Panel, MultiSelect, Select
from bokeh.palettes import Spectral4, Spectral8, Viridis6, Viridis11
from bokeh.plotting import figure, output_file, show
import itertools
from math import sqrt

from bs4 import BeautifulSoup
from urllib.request import urlopen

import sys, json
import os

output_file('indexnodelink.html')

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
#   NLD_processing_graph
###############################################################################
def NLD_pocessing_graph(g, weights, colors, layout):
    if layout != nx.random_layout:
        graph = from_networkx(g, layout, scale=1, center=(0,0))
    else:
        graph = from_networkx(g, layout, center=(0,0))

    # nodes and egdes attributes
    graph.node_renderer.data_source.data['degree'] = list(zip(*g.degree))[1]
    graph.edge_renderer.data_source.data['weight'] = weights
    graph.edge_renderer.data_source.add(colors, 'color')

    graph.node_renderer.glyph            = Circle(size='nodesize', fill_alpha=0.8, fill_color='royalblue')
    graph.node_renderer.selection_glyph  = Circle(size=10, fill_alpha=0.8, fill_color='red')
    graph.node_renderer.hover_glyph      = Circle(size=10, fill_alpha=0.8, fill_color='yellow')

    graph.edge_renderer.glyph            = MultiLine(line_width=3, line_alpha=0.8, line_color='color')
    graph.edge_renderer.selection_glyph  = MultiLine(line_width=4, line_alpha=0.8, line_color='red')
    graph.edge_renderer.hover_glyph      = MultiLine(line_width=4, line_alpha=0.8, line_color='yellow')
    graph.edge_renderer.glyph.line_width = {'field': 'weight'}

    graph.selection_policy = NodesAndLinkedEdges()
    graph.inspection_policy = NodesAndLinkedEdges()

    return graph


###############################################################################
#   NLD_FD_processing_graph - ForceDirected
###############################################################################
def NLD_FD_pocessing_graph(g, weights, colors, layout, degree_sequence):
    magicnumber = (g.number_of_nodes()/degree_sequence[0]**2)
    r= 2.2/magicnumber
    my_points=nx.fruchterman_reingold_layout(g, k = r, iterations=100)
    graph_fd = from_networkx(g, my_points)

    #posxy=nx.fruchterman_reingold_layout(g)
    #for i in range(len(posxy()):
    #    posxy
    #radii = np.random.random(size=N) * 1.5
    my_colors = []
    for key, value in my_points.items():
        x = value[0] + 1.0 # x = -1 .. +1, so move to 0 ... 2
        y = value[1] + 1.0 # y = -1 .. +1, so move to 0 ... 2
        my_colors.append( "#%02x%02x%02x" % (int(50+100*x), int(30+100*y), 150) )

    # nodes and egdes attributes
    graph_fd.node_renderer.data_source.data['degree'] = list(zip(*g.degree))[1]
    graph_fd.node_renderer.data_source.data['degree2'] = [sqrt(x+6) for x in graph_fd.node_renderer.data_source.data['degree']]
    graph_fd.node_renderer.data_source.data['nodessize'] = [x*115/(sqrt(g.number_of_nodes()+(np.mean(degree_sequence)))) for x in graph_fd.node_renderer.data_source.data['degree2']]

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

    return graph_fd


###############################################################################
#   NLD_add_tools
###############################################################################
def NLD_add_tools(plot):
    plot.add_tools(WheelZoomTool())
    plot.add_tools(ZoomInTool())
    plot.add_tools(ZoomOutTool())
    plot.add_tools(ResetTool())
    plot.add_tools(UndoTool())
    plot.add_tools(RedoTool())
    plot.add_tools(PanTool())
    plot.add_tools(TapTool())
    plot.add_tools(SaveTool())
    plot.add_tools(BoxSelectTool())
    plot.add_tools(LassoSelectTool())
    plot.add_tools(BoxZoomTool())
    # !!! Hover the node attributes !!!
    node_hover = HoverTool(tooltips=[('Name', '@index'), ('Degree', '@degree'), #('Node Size', '@nodesize'),
                                    ('Min Weight', '@minweight'), ('Max Weight', '@maxweight'),
                                    ('Average Weight', '@avrweight'), ('Sum Weight', '@sumweight')])
    plot.add_tools(node_hover)


###############################################################################
#   NLD_processing
###############################################################################
def NLD_processing(df):
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

    # Making a function to map color to edges
    color_palette = list(reversed(Viridis11[:8]))
    w_max = df.values.max()
    w_min = df.values.min()
    step = (w_max-w_min)/(len(color_palette)-1)


    colors = []
    # Create a graph with 1-way edges for faster painting
    g=nx.DiGraph()
    for row in df.index.values:
        g.add_node(row)
        for column in df.index.values:
            if  row < column:
                if (df[row][column] > 0):
                    color_index = int((df[row][column] - w_min) / step)
                    g.add_edge(row, column, weight=df[row][column], color=color_palette[color_index])
                    colors.append(color_palette[color_index])

    weights = []
    # Create a separate graph with 2-way edges only to calculate weights
    g_w=nx.DiGraph()
    for row in df.index.values:
        g_w.add_node(row)
        for column in df.index.values:
            if  row != column:
                if (df[row][column] > 0):
                    g_w.add_edge(row, column, weight=df[row][column], color=color_palette[color_index])
                    weights.append(df[row][column])

    # do not draw edges with different widths if the max weight is too big
    if max(weights) > 30:
        for index, w in enumerate(weights):
            weights[index] = 1

    # loop over all nodes to find neighbors and set min, max, sum for egdes weights connected to a node
    node_w_dict = {}
    for n in list_columns_int:
        node_weight_list = []
        for nb in nx.neighbors(g_w, n):
            node_weight_list.append(nx.get_edge_attributes(g_w,'weight')[n, nb])
        len_list = len(node_weight_list)
        if len_list != 0:
            node_min_weight = min(node_weight_list)
            node_max_weight = max(node_weight_list)
            node_sum_weight = sum(node_weight_list)
            node_avr_weight = node_sum_weight / len_list
        else:
            node_min_weight = 0
            node_max_weight = 0
            node_sum_weight = 0
            node_avr_weight = 0
        node_w_dict.update({n:{'minweight':node_min_weight, 'maxweight':node_max_weight, 'avrweight':node_avr_weight, 'sumweight':node_sum_weight}})
    nx.set_node_attributes(g, node_w_dict)

    # Making a function to map node size
    deg_node_size_list = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    deg_max = max(list(list(zip(*g.degree))[1]))
    deg_min = min(list(list(zip(*g.degree))[1]))
    deg_step = (deg_max-deg_min)/(len(deg_node_size_list)-1)
    i = 0
    node_s_dict = {}
    for node in list(list(zip(*g.degree))[0]):
        deg_index = int((list(list(zip(*g.degree))[1])[i] - deg_min) / deg_step)
        node_s_dict.update({node:{'nodesize':deg_node_size_list[deg_index]}})
        i += 1
    nx.set_node_attributes(g, node_s_dict)


    # create a dictoinary with double for loop
    mapping = {old_label:new_label for old_label, new_label in itertools.zip_longest(sorted(g.nodes()), list_columns_names, fillvalue=1)}

    # relabel the names of the nodes from integers back to strings
    nx.relabel_nodes(g, mapping, copy=False)


    # Organize common layouts' size for NLD
    NLD_width  = 780
    NLD_height = 690

    color_mapper = LinearColorMapper(palette=color_palette, low=w_min, high=w_max)
    color_bar = ColorBar(color_mapper = color_mapper, border_line_color = None, location = (0,0))

    # circular layout
    plot_circle = Plot(plot_width=NLD_width, plot_height=NLD_height,
                x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))

    graph_circle = NLD_pocessing_graph(g, weights, colors, nx.circular_layout)

    NLD_add_tools(plot_circle)

    plot_circle.add_layout(color_bar, 'right')

    plot_circle.renderers.append(graph_circle)


    # spring layout
    plot_spring = Plot(plot_width=NLD_width, plot_height=NLD_height,
                x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))

    graph_spring = NLD_pocessing_graph(g, weights, colors, nx.spring_layout)

    NLD_add_tools(plot_spring)

    plot_spring.add_layout(color_bar, 'right')

    plot_spring.renderers.append(graph_spring)


    # force-directed layout
    plot_fd = Plot(plot_width=NLD_width, plot_height=NLD_height,
                x_range=Range1d(-1.1, 1.1), y_range=Range1d(-1.1, 1.1))

    degree_sequence = sorted([d for n, d in g.degree()], reverse=True)
    magicnumber = (g.number_of_nodes()/degree_sequence[0]**2)
    r = 2.2/magicnumber
    my_points=nx.fruchterman_reingold_layout(g, iterations=100)
    graph_fd = NLD_FD_pocessing_graph(g, weights, colors, nx.fruchterman_reingold_layout(g,k=r, iterations=0, pos=my_points, scale=1, center=(0,0)), degree_sequence)

    NLD_add_tools(plot_fd)

    plot_fd.add_layout(color_bar, 'right')

    plot_fd.renderers.append(graph_fd)


    # random layout
    plot_random = Plot(plot_width=NLD_width, plot_height=NLD_height,
                x_range=Range1d(-0.1, 1.1), y_range=Range1d(-0.1, 1.1))

    graph_random = NLD_FD_pocessing_graph(g, weights, colors, nx.random_layout, degree_sequence)

    NLD_add_tools(plot_random)

    plot_random.add_layout(color_bar, 'right')

    plot_random.renderers.append(graph_random)

    # Create panels for each layout
    circle_panel = Panel(child=plot_circle, title='Circle layout')
    spring_panel = Panel(child=plot_spring, title='Spring layout')
    random_panel = Panel(child=plot_random, title='Random layout')
    fd_panel     = Panel(child=plot_fd,     title='Force-Directed layout')

    # Assign NLD panels to Tabs
    tabsNLD_int = Tabs(tabs=[circle_panel, spring_panel, fd_panel, random_panel])
    return tabsNLD_int


###############################################################################
#   main
###############################################################################
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

    df_full = file_processing(filename)

    # subset dataframes
#    df_subset = df_full.loc["Jim Thomas":"James Landay", "Jim Thomas":"James Landay"] # 50
    df_subset = df_full.loc["Jim Thomas":"Carl Burnham", "Jim Thomas":"Carl Burnham"] # 250

    tabsNLD = NLD_processing(df_subset)

    # Show the tabbed layout
    show(tabsNLD)

##############################################################################
#   Start process
###############################################################################
if __name__ == '__main__':
    main()


# Fetch the html file
response = urlopen('file://indexnodelink.html')
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
