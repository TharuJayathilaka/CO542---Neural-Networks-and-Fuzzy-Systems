from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from itertools import product
import time

#-------------------------Question 1-----------------------------------------------------------------------------------------------------------------------------------------------

data = np.array([[1,4],[3,7],[3,2],[4,1]])
labels = [0, 1, 1, 0]

#-------------------------Question 2-----------------------------------------------------------------------------------------------------------------------------------------------

plt.scatter([point[0] for point in data], [point[1] for point in data], c = labels)
plt.show()

classifier = Perceptron(max_iter = 40)

#-------------------------Question 3-----------------------------------------------------------------------------------------------------------------------------------------------

#training time
start = time.time()
classifier.fit(data, labels)
stop = time.time()
print(f"Training time: {stop - start}s")
print(classifier.score(data, labels))


#-------------------------Question 4-----------------------------------------------------------------------------------------------------------------------------------------------

data1 = np.array([[1,4],[3,7],[3,2],[53,35]])
labels1 = [0, 1, 1, 0]

#-------------------------Question 5-----------------------------------------------------------------------------------------------------------------------------------------------

plt.scatter([point[0] for point in data1], [point[1] for point in data1], c = labels1)
plt.show()

#-------------------------Question 6-----------------------------------------------------------------------------------------------------------------------------------------------

#training time
start1 = time.time()
classifier.fit(data1, labels1)
stop1 = time.time()
print(f"Training time: {stop1 - start1}s")
print(classifier.score(data1, labels1))

#-------------------------Question 7-----------------------------------------------------------------------------------------------------------------------------------------------

new = [7,14] 
print(classifier.predict([new]))

plt.scatter([point[0] for point in data1], [point[1] for point in data1], c = labels1)
plt.scatter([new[0]],[new[1]], c = 'r')
plt.show()

#-------------------------Question 8  /  Question 9  ------------------------------------------------------------------------------------------------------------------------------

# first define the step size
s = 0.1 

x_l  = data1[:, 0].min() - 1
x_r  = data1[:, 0].max() + 1
y_l  = data1[:, 1].min() - 1
y_r  = data1[:, 1].max() + 1

x = np.arange(x_l, x_r, s)
y = np.arange(y_l, y_r, s)
x,y = np.meshgrid(x,y)

fig1,fig2 = plt.subplots()

z = classifier.predict(np.c_[x.ravel(), y.ravel()])
z = z.reshape(x.shape)

new_plot = fig2.contourf(x, y, z, cmap=plt.get_cmap('Spectral'))
plt.scatter([point[0] for point in data1], [point[1] for point in data1], c = labels)
plt.scatter([new[0]],[new[1]], c = 'r')
plt.show()


