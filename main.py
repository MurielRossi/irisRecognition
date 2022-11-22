import shipy
import numpy
import matplotlib
import pandas as pd
import sklearn
from matplotlib import pyplot
from pandas.plotting import scatter_matrix

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# shape
print(dataset.shape)

# head
print(dataset.head(20)) #default, the first 5 instances

# descriptions
#count - The number of not-empty values.
#mean - The average (mean) value.
# std - The standard deviation.
#min - the minimum value.
#*Percentile meaning: how many of the values are less than the given percentile.
#max - the maximum value.
print(dataset.describe())

# class distribution: how much instances x class
print(dataset.groupby('class').size())

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# histograms
dataset.hist()
pyplot.show()

# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()



