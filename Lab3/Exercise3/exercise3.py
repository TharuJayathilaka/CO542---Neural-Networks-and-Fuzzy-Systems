import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import Perceptron
from mpl_toolkits.mplot3d import Axes3D
from skimage import io
from skimage.transform import rescale

trainSet = './classpics/train/'
testSet = './classpics/test/'
train = os.listdir(trainSet)
test = os.listdir(testSet)
trainListOld = []
testListOld = []
trainListNew = []
testListNew = []

for train_file in train:
    image_old = io.imread(trainSet+train_file)
    trainListOld.append(image_old)
	#convert to Gray while reading the image
    grayImage = io.imread(trainSet+train_file, as_gray=True) 
    grayImage = rescale(grayImage, scale=(0.5, 0.5))
    trainListNew.append(grayImage)  

for test_file in test:
    image_old = io.imread(testSet+test_file)
    testListOld.append(image_old)
	#convert to Gray while reading the image
    grayImage = io.imread(testSet+test_file, as_gray=True)
    grayImage = rescale(grayImage, scale=(0.5, 0.5))
    testListNew.append(grayImage)

trainingImages = np.array(trainListNew)
testingImages = np.array(testListNew)

# There are two types of classes. Student has a beard and Student has not a beard 
# student has not a beard = 0
# student has a beard = 1
trainingImageLabels = np.array([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
subset_name = ["Student has not a beard","Student has a beard"]

n = 0
for i in trainListOld:
    print(subset_name[trainingImageLabels[number]])
    n = n + 1

sample_nums, numxx, numyy = trainingImages.shape
trained_images = trainingImages.reshape((sample_nums,numxx*numyy))
classifier = Perceptron(max_iter = 40)
classifier.fit(trained_images, trainingImageLabels) 
print(classifier.score(trained_images, trainingImageLabels))


sample_nums, numxx, numyy = testingImages.shape
tested_images = testingImages.reshape((sample_nums,numxx*numyy))
predictions = classifier.predict(tested_images)
print(classifier.predict(tested_images)) 
    
