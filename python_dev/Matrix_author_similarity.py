import numpy as np
import pandas as pd
import statsmodels.api as sm
import sqlite3
import csv
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
with open('GephiMatrix_author_similarity.csv', 'r') as r, open('GephiMatrix_author_similarity1.csv', 'w') as w:
    for num, line in enumerate(r):
        if num >= 0:
            if line[0] == "\"" and line[-1] == "\"":
                newline = line[1: -1]
            elif line[0] == "\"" and line[-1] == "\n" and line[-2] == "\"":
                newline = line[1: -2] + "\n"
            else:
                newline = line
        else:
            newline = line
        w.write(newline)
with open('GephiMatrix_author_similarity1.csv', 'r') as r, open('GephiMatrix_author_similarity2.csv', 'w') as w:
    for num, line in enumerate(r):
        if num >= 0:
            if line[0] == ";":
                newline = line[1:]
            else:
                newline = line
        else:
            newline = line
        w.write(newline)
with open('GephiMatrix_author_similarity2.csv', 'r') as r, open('GephiMatrix_author_similarity3.csv', 'w') as w:
    for num, line in enumerate(r):
        if num >= 0:
            if line[-1] == ';':
                newline = line[:-1]
            elif line[-1] == "\n" and line[-2] == ";":
                newline = line[: -2] + "\n"
            else:
                newline = line
        else:
            newline = line
        w.write(newline)
df = pd.read_csv('GephiMatrix_author_similarity3.csv', sep = ';')
df.head()
df_new = df.loc["Jim_Thomas":"Jeffrey_Brainerd", "Jim_Thomas":"Jeffrey_Brainerd"]
df_new
matrix = np.matrix(df_new)
matrix
names = df_new.index.values
names
cax = plt.matshow(matrix, vmin = 0, vmax = 1)
x_pos = np.arange(len(names))
plt.xticks(x_pos, names, rotation = 90)
y_pos = np.arange(len(names))
plt.yticks(y_pos, names)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)
fig.colorbar(cax)


plt.show()