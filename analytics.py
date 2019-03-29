import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize
#from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import statsmodels.api as sm
#from sklearn.linear_model import LinearRegression
from sklearn import linear_model
#import statsmodels.formula.api as sm
#import datetime
#from dateutil.parser import parse

data_raw = pd.read_csv('<name of dataset>.csv', skipinitialspace = True)
#data_raw = np.genfromtxt('<filename.txt>')
#niose = np.genfromtxt('<filename.txt>')
#import the raw noise as well

plt.figure()
plt.plot([1,2],[3,4])
plt.show()

plt.__version__
data = data_raw - noise
#show the first couple rows to the raw dataself.
data_raw.head()
data.head()
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))

#plot a hist of the raw data to get a look at the distribution, look for any erroneous points/outliers
plt.figure()
plt.hist(data_raw)
plt.title("Raw Data")
plt.xlabel("X label")
plt.ylabel("Y label")
plt.show()

plt.figure()
plt.hist(data)
plt.title("Data without noise")
plt.xlabel("X label")
plt.ylabel("Y label")
plt.show()

#look at a scatter plot of the data, look for any erroneous points/outliers
plt.figure()
plt.scatter(<data_raw_1>, <data_raw_2>)
plt.title("Raw Data")
plt.xlabel("X label")
plt.ylabel("Y label")
plt.show()

#set the bounds on the data set for cleaning and repeat for all columns
data_raw["column_1"] = data_raw["column_1"][<lower_bound> <= data_raw["column_1"] <= <upper_bound>]
#might need to create ranges and seperate the dataset

def chi_sq(data_1, data_true, bin):
    x=[]
    for i in range(len(data_1)):
        z[i] = (data_1[i] - data_true[i])**2/data_true[i]
        x.append(z[i])
    return sum(x)/(bin-1)


def chi_sq_sig(data_1, data_true, sigma, bin):
    x=[]
    for i in range(len(data_1)):
        z[i] = (data_1[i] - data_true[i])**2/sigma[i]
        x.append(z[i])
    return sum(x)/(bin-1)

def mean(data):
    return sum(data)/len(data)

def sigma(data):
    x = []
    N = len(data)
    x_bar = sum(data)/len(data)
    for i in range(len(data)):
        d = (data[i] - x_bar)**2
        x.append(d)
    return np.sqrt(sum(x)/(N-1))

#data = np.genfromtxt('file_path.csv', delimiter=',')

#define the function that is believed to fit curve... Gaussian
def func(r,a,b,c):
    return a + b*np.exp(c*r**2)

pop, pcov = optimize.curve_fit(func, xdata, ydata)
(a,b,c) = pop
print('a =',a,'b =',b,'c =',c)

func = func(xdata,a,b,c)

plt.figure()
plt.plot(xdata, func)
plt.show()

help(plt)

import seaborn as sns
tips = sns.load_dataset("tips")
sns.lineplot(x="total_bill", y="tip", data=tips)
sns.boxplot()
from bokeh.plotting import figure, output_file, show

# prepare some data
x = [1, 2, 3, 4, 5]
y = [6, 7, 2, 4, 5]


import warnings

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

import matplotlib._layoutbox as layoutbox

plt.rcParams['savefig.facecolor'] = "0.8"
plt.rcParams['figure.figsize'] = 4.5, 4.


def example_plot(ax, fontsize=12, nodec=False):
    ax.plot([1, 2])

    ax.locator_params(nbins=3)
    if not nodec:
        ax.set_xlabel('x-label', fontsize=fontsize)
        ax.set_ylabel('y-label', fontsize=fontsize)
        ax.set_title('Title', fontsize=fontsize)
    else:
        ax.set_xticklabels('')
        ax.set_yticklabels('')


fig, ax = plt.subplots()
example_plot(ax, fontsize=24)

# output to static HTML file
#output_file("lines.html")

# create a new plot with a title and axis labels
p = figure(title="simple line example", x_axis_label='x', y_axis_label='y')

# add a line renderer with legend and line thickness
p.line(x, y, legend="Temp.", line_width=2)

# show the results
show(p)
