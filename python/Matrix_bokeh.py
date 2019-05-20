
# coding: utf-8

# In[15]:

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from bokeh.io import output_file, output_notebook
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, LogTicker
from bokeh.layouts import row, column, gridplot
from bokeh.models.widgets import Tabs, Panel
from bokeh.charts import HeatMap, output_file, show
from bokeh.models import HoverTool
from bokeh.palettes import Magma6

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
        # removes all "_" in the names of the people and replaves them by " " (spacebar)
        newline = newline.replace("_", " ")
        w.write(newline)

df = pd.read_csv(filename + '_03' + fileext, sep=separator) 
df.drop(df[df.index.duplicated()])
df_new = df.loc["Jim_Thomas":"Jeffrey_Brainerd", "Jim_Thomas":"Jeffrey_Brainerd"]
counts = df_new.values
names = df_new.index.values.tolist()
names1 = df_new.index.values.tolist()
names.extend(names1 * (len(names1) - 1))
def duplicate(names1, n):
    return [ele for ele in names1 for _ in range(n)]
xname = duplicate(names1, len(names1))
yname = names


alpha = []

for i, name in enumerate(names1):
    for j, name1 in enumerate(names1):
        alpha.append(min(counts[i, j]/4, 0.9) + 0.1)



data=dict(
    xname=xname,
    yname=yname,
    count=counts,
    alphas=alpha,
)

p = figure(title="Matrix Author Similarity",
           x_axis_location="above",
           x_range=list(reversed(names1)), y_range=names1)
hover = HoverTool(tooltips = [('Names', '@yname, @xname'), ('Value', '@count')])
p.add_tools(hover)
p.plot_width = 1000
p.plot_height = 1000
p.grid.grid_line_color = None
p.axis.axis_line_color = None
p.axis.major_tick_line_color = None
p.axis.major_label_text_font_size = "5pt"
p.axis.major_label_standoff = 0
p.xaxis.major_label_orientation = np.pi/3

p.rect('xname', 'yname', 0.9, 0.9, source=data, line_color=None,
       hover_line_color='black', alpha = 'alphas')

mapper = LinearColorMapper(palette=Magma6, low=0, high=1)
color_bar = ColorBar(color_mapper = mapper, border_line_color = None, location = (0,0))

data1=dict(
    xname=xname,
    yname=yname,
    count=counts,
)

plot = figure(title="Matrix Author Similarity",
           x_axis_location="above",
           x_range=list(reversed(names1)), y_range=names1, toolbar_location = 'below')
hover = HoverTool(tooltips = [('Names', '@yname, @xname'), ('Value', '@count')])
plot.add_tools(hover)
plot.plot_width = 1000
plot.plot_height = 1000
plot.grid.grid_line_color = None
plot.axis.axis_line_color = None
plot.axis.major_tick_line_color = None
plot.axis.major_label_text_font_size = "5pt"
plot.axis.major_label_standoff = 0
plot.xaxis.major_label_orientation = np.pi/3

plot.rect('xname', 'yname', 0.9, 0.9, source=data1, line_color=None,
       hover_line_color='black', fill_color={'field': 'count', 'transform': mapper})
plot.add_layout(color_bar, 'right')

alpha_panel = Panel(child = p, title = 'Alpha model')
color_panel = Panel(child = plot, title = 'Color model')
tabs = Tabs(tabs = [alpha_panel, color_panel])
      
output_file("Matrix.html")

show(tabs) # show the plot

