from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from sklearn.neural_network import MLPRegressor

# generate data for input
x = np.arange(-100, 100, 0.001)

# by using generated data for x, generate data for output
y = -x**4 + x**3 + 23*x**2 - 21*x + 32

# get the size of the input and output vectors
print(x.shape)
print(y.shape)

#generate the graph
plt.scatter(x, y ,c='r')
plt.show()

#Model the MLP using MLPRegressor instead of MLPClassifier
x = x.reshape(-1,1)
model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=100)
model.fit(x,y)
print("-------------------Model score-----------------")
print(model.score(x,y))

#Generate a new test set and test it with the network.
x1 = np.arange(-100, 100, 10)
x1=x1.reshape(-1, 1)
y1 = model.predict(x1)

#Plot the train data and model predictions on a same plot
plt.scatter(x, y ,c='b', marker="s", label='Training')
plt.scatter(x1, y1, c='r', marker="o", label='Predictions')
plt.show()
