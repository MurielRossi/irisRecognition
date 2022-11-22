import shipy
import numpy
import matplotlib
import pandas as pd
import sklearn

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pd.read_csv(url, names=names)

# shape
print(dataset.shape)

# head
print(dataset.head(20)) #default, the first 5 instances





