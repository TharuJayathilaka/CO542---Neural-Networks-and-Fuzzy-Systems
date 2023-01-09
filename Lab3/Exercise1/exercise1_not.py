from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

#create a variable named data that is a list that contains thhe four possible inputs to an AND gate

data = np.array([[0], [1]])
data = data.reshape(-1,1)
labels = [1,0]

plt.scatter(data,labels, c=labels )
#the third parameter "c=labels" will make the points with label 1 a dfferent color than points with label 0.
plt.show()

#let's build a perceptron to learn AND

classifier= Perceptron(max_iter=40)
classifier.fit(data,labels)
print(classifier.score(data,labels))
print(classifier.predict(data))

x3 = np.arange(0, 1, 0.01)
y3 = np.arange(0, 1, 0.01)

x3, y3 = np.meshgrid(x3, y3) 

points=np.zeros((x3.shape[0],x3.shape[1]))
for i in range(x3.shape[0]):
    for j in range(y3.shape[0]):
        points[i][j] = classifier.predict([[x3[i][j]]])
    
    
fig = plt.figure() 
axes = fig.gca(projection ='3d',title='NOT Logic Gate Surface Plot') #3d projection to get output surface       
axes.plot_surface(x3, y3, points)  
plt.show()