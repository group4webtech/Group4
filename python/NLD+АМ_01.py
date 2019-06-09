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
from bokeh.palettes import Spectral4, Spectral8, Viridis6, Viridis11
from bokeh.plotting import figure, output_file, show
import itertools

from math import sqrt
from bs4 import BeautifulSoup
from urllib.request import urlopen
import sys, json


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

    with open(filename, 'r') as r, open('DBL_' + filename, 'w') as w:
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

    # dataframe for NLD
    df_int = pd.read_csv('DBL_' + filename, sep=separator)     # full csv file with 1053x1053 values
    return df_int


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
def AM_processing(df_new):
    max_value = df_new.values.max()
    min_value = df_new.values.min()
    values = df_new.values
    counts = values.astype(float)
    #counts = df_new.values
    names  = df_new.index.values.tolist()
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
#   NLD_processing_graph
###############################################################################
def NLD_pocessing_graph(g, weights, colors, layout):
    graph = from_networkx(g, layout, scale=1, center=(0,0))

    # nodes and egdes attributes
    graph.node_renderer.data_source.data['degree'] = list(zip(*g.degree))[1]
    graph.edge_renderer.data_source.data['weight'] = weights
    graph.edge_renderer.data_source.add(colors, 'color')

    graph.node_renderer.glyph            = Circle(size=10, fill_alpha=0.8, fill_color='royalblue')
    graph.node_renderer.selection_glyph  = Circle(size=10, fill_alpha=0.8, fill_color='red')
    graph.node_renderer.hover_glyph      = Circle(size=10, fill_alpha=0.8, fill_color='yellow')
#    graph.node_renderer.glyph.color      = {'field': ''}

    graph.edge_renderer.glyph            = MultiLine(line_width=3, line_alpha=0.8, line_color='color')
    graph.edge_renderer.selection_glyph  = MultiLine(line_width=4, line_alpha=0.8, line_color='red')
    graph.edge_renderer.hover_glyph      = MultiLine(line_width=4, line_alpha=0.8, line_color='yellow')
    graph.edge_renderer.glyph.line_width = {'field': 'weight'}
#    graph.edge_renderer.glyph.line_color = {'field': 'color'}

    graph.selection_policy = NodesAndLinkedEdges()
    graph.inspection_policy = NodesAndLinkedEdges()

    return graph


###############################################################################
#   NLD_FD_processing_graph - ForceDirected
###############################################################################
def NLD_FD_pocessing_graph(g, weights, colors):
    r = 1.8*(max(g.degree())[1])/sqrt(g.number_of_nodes())

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

    # nodes and egdes attributes
    graph_fd.node_renderer.data_source.data['degree'] = list(zip(*g.degree))[1]
    graph_fd.node_renderer.data_source.data['degree2'] = [x+1 for x in graph_fd.node_renderer.data_source.data['degree']]
    graph_fd.node_renderer.data_source.data['my_fill_color'] = my_colors
    graph_fd.edge_renderer.data_source.data['weight'] = weights
    graph_fd.edge_renderer.data_source.add(colors, 'color')

    graph_fd.node_renderer.glyph            = Circle(size ="degree2",  fill_color='my_fill_color', fill_alpha=0.85)
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
    plot.add_tools(ZoomOutTool())
    plot.add_tools(ZoomInTool())
    plot.add_tools(ResetTool())
    plot.add_tools(UndoTool())
    plot.add_tools(RedoTool())
    plot.add_tools(PanTool())
    plot.add_tools(TapTool())
    plot.add_tools(SaveTool())
    plot.add_tools(BoxSelectTool())
    plot.add_tools(LassoSelectTool())
    # !!! Hover the node attributes !!!
    node_hover = HoverTool(tooltips=[('Name', '@index'), ('Degree', '@degree'), ('Max Weight', '@maxweight')])
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
        for column in df.index.values:
            if  row < column:
                if (df[row][column] > 0):
                    color_index = int((df[row][column] - w_min) / step)
                    g.add_edge(row, column, weight=df[row][column], color=color_palette[color_index])
                    weights.append(df[row][column])
                    colors.append(color_palette[color_index])
                else:
                     g.add_node(row)

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

    graph_fd = NLD_FD_pocessing_graph(g, weights, colors)

    NLD_add_tools(plot_fd)

    plot_fd.add_layout(color_bar, 'right')

    plot_fd.renderers.append(graph_fd)


    # Create panels for each layout
    circle_panel = Panel(child=plot_circle, title='Circle layout')
    spring_panel = Panel(child=plot_spring, title='Spring layout')
    fd_panel     = Panel(child=plot_fd, title='Force-Directed layout')

    # Assign NLD panels to Tabs
    tabsNLD_int = Tabs(tabs=[circle_panel, spring_panel, fd_panel])
    return tabsNLD_int


###############################################################################
#   main
###############################################################################
def main():
    global filename
    # Sum  of all the items in the providen array
    total_sum_inArray = 0

    #get our data as an array from read_in()
#    lines = read_in()
#    filename = "./upload/" + lines[0]
    filename = 'DBL.csv'
    #print(filename)
    #return the sum to the output stream

    df = file_processing(filename)

    output_file('index.html')

    # subset dataframe for AM
    df_subset = df.loc["Jim Thomas":"James Landay", "Jim Thomas":"James Landay"]

    tabsAM = AM_processing(df_subset)

    tabsNLD = NLD_processing(df)

    grid = gridplot([[tabsNLD, tabsAM]])

    # Show the tabbed layout
    show(grid)


    # Fetch the html file
    response = urlopen('file://index.html')
    html_output = response.read()

    # Parse the html file
    soup = BeautifulSoup(html_output, 'html.parser')

    # Format the parsed html file
    strhtm = soup.prettify()

    f = open('views/graphs' + '.ejs','w')

    message = """{strhtm}
    """.format(strhtm=strhtm)


##############################################################################
#   Start process
###############################################################################
if __name__ == '__main__':
    main()
