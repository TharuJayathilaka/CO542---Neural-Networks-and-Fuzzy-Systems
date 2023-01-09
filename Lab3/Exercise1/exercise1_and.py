from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

#create a variable named data that is a list that contains thhe four possible inputs to an AND gate

data=[[0,0],[0,1],[1,0],[1,1]]
labels=[0,0,0,1]

plt.scatter([point[0] for point in data],[point[1] for point in data], c=labels )
#the third parameter "c=labels" will make the points with label 1 a dfferent color than points with label 0.
plt.show()

#let's build a perceptron to learn AND

classifier= Perceptron(max_iter=40)
classifier.fit(data,labels)
print(classifier.score(data,labels))
print(classifier.predict(data))

x1 = np.arange(0, 1, 0.01)
y1 = np.arange(0, 1, 0.01)

x1, y1 = np.meshgrid(x1, y1) 

points=np.zeros((x1.shape[0],x1.shape[1]))
for i in range(x1.shape[0]):
    for j in range(y1.shape[0]):
        points[i][j] = classifier.predict([[x1[i][j],y1[i][j]]])
    
# Output surface using 3d projection     
fig = plt.figure() 
axes = fig.gca(projection ='3d',title='AND Logic Gate Surface Plot')       
axes.plot_surface(x1, y1, points)        
plt.show()