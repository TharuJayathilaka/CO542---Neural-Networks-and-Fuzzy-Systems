from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

#create a variable named data that is a list that contains thhe four possible inputs to an AND gate

data = [[0,0], [0,1],[1,0],[1,1]]
labels = [0,1,1,0]

plt.scatter([point[0] for point in data], [point[1] for point in data], c= labels)
#the third parameter "c=labels" will make the points with label 1 a dfferent color than points with label 0.
plt.show()

#Creating a network‘netXOR’ with 2 neurons in the input layer, 5 neurons in the hidden layer and 1 output neuron
netXOR = MLPClassifier(hidden_layer_sizes = (5), activation = 'relu', random_state = 1)
netXOR.fit(data,labels)
print("score of netXOR")
print(netXOR.score(data,labels))