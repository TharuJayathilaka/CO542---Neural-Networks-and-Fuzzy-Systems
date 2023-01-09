import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

#-----------------------------Question 1-----------------------------------------------------------------------------------------------------------------------------------------

iris_data = pd.read_csv("iris.csv")
iris_data.head()
iris_data = iris_data.drop(['Id'], 1)
iris_data.head()

#-----------------------------Question 2-----------------------------------------------------------------------------------------------------------------------------------------

sns.pairplot(iris_data)
plt.show()

#-----------------------------Question 3-----------------------------------------------------------------------------------------------------------------------------------------

feature_names = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
features = iris_data[feature_names]
labels = iris_data['Species']
print(features.shape)
print(labels.shape)

#-----------------------------Question 4-----------------------------------------------------------------------------------------------------------------------------------------

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(labels)
labels = label_encoder.transform(labels)

#-----------------------------Question 5-----------------------------------------------------------------------------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2,random_state=0)
print(x_train.shape)
print(y_train.shape)

#-----------------------------Question 6-----------------------------------------------------------------------------------------------------------------------------------------

s_f = StandardScaler()
x_train = s_f.fit_transform(x_train)
x_test = s_f.transform(x_test)

#-----------------------------Question 7-----------------------------------------------------------------------------------------------------------------------------------------

model = MLPClassifier(hidden_layer_sizes=(32), activation="relu", random_state=1, max_iter=50)

#-----------------------------Question 8-----------------------------------------------------------------------------------------------------------------------------------------

result = model.fit(x_train, y_train)
l_pred = result.predict(x_test)
print(l_pred)
print(result.score(x_test, y_test))

#-----------------------------Question 9-----------------------------------------------------------------------------------------------------------------------------------------
target_names = ["Setosa", "Versicolor", "Virginica"]
fig = plot_confusion_matrix(result, x_test, y_test, display_labels=target_names)
fig.figure_.suptitle("Confusion Matrix for Iris Dataset")
plt.show()

#-----------------------------Question 10-----------------------------------------------------------------------------------------------------------------------------------------

print(classification_report(y_test, l_pred, target_names=["Setosa", "Versicolor", "Virginica"]))

#-----------------------------Question 11-----------------------------------------------------------------------------------------------------------------------------------------

#max_iter=100
model = MLPClassifier(hidden_layer_sizes=(32), activation="relu", random_state=1, max_iter=100)
result = model.fit(x_train, y_train)
l_pred = result.predict(x_test)
print(l_pred)
print(result.score(x_test, y_test))
fig = plot_confusion_matrix(result, x_test, y_test, display_labels=target_names)
fig.figure_.suptitle("Confusion Matrix for Iris Dataset")
plt.show()
print(classification_report(y_test, l_pred, target_names=target_names))


#max_iter=300
model = MLPClassifier(hidden_layer_sizes=(32), activation="relu", random_state=1, max_iter=300)
result = model.fit(x_train, y_train)
l_pred = result.predict(x_test)
print(l_pred)
print(result.score(x_test, y_test))
fig = plot_confusion_matrix(result, x_test, y_test, display_labels=target_names)
fig.figure_.suptitle("Confusion Matrix for Iris Dataset")
plt.show()
print(classification_report(y_test, l_pred, target_names=target_names))


#max_iter=500
model = MLPClassifier(hidden_layer_sizes=(32), activation="relu", random_state=1, max_iter=500)
result = model.fit(x_train, y_train)
l_pred = result.predict(x_test)
print(l_pred)
print(result.score(x_test, y_test))
fig = plot_confusion_matrix(result, x_test, y_test, display_labels=target_names)
fig.figure_.suptitle("Confusion Matrix for Iris Dataset")
plt.show()
print(classification_report(y_test, l_pred, target_names=target_names))


#-----------------------------Question 12-----------------------------------------------------------------------------------------------------------------------------------------

#test_size=0.1
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.1,random_state=0)
print(x_train.shape)
print(y_train.shape)
s_f = StandardScaler()
x_train = s_f.fit_transform(x_train)
x_test = s_f.transform(x_test)
model = MLPClassifier(hidden_layer_sizes=(32), activation="relu", random_state=1, max_iter=100)
result = model.fit(x_train, y_train)
l_pred = result.predict(x_test)
print(l_pred)
print(result.score(x_test, y_test))
fig = plot_confusion_matrix(result, x_test, y_test, display_labels=target_names)
fig.figure_.suptitle("Confusion Matrix for Iris Dataset")
plt.show()
print(classification_report(y_test, l_pred, target_names=target_names))


#test_size=0.3
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3,random_state=0)
print(x_train.shape)
print(y_train.shape)
s_f = StandardScaler()
x_train = s_f.fit_transform(x_train)
x_test = s_f.transform(x_test)
model = MLPClassifier(hidden_layer_sizes=(32), activation="relu", random_state=1, max_iter=100)
result = model.fit(x_train, y_train)
l_pred = result.predict(x_test)
print(l_pred)
print(result.score(x_test, y_test))
fig = plot_confusion_matrix(result, x_test, y_test, display_labels=target_names)
fig.figure_.suptitle("Confusion Matrix for Iris Dataset")
plt.show()
print(classification_report(y_test, l_pred, target_names=target_names))


#test_size=0.5
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.5,random_state=0)
print(x_train.shape)
print(y_train.shape)
s_f = StandardScaler()
x_train = s_f.fit_transform(x_train)
x_test = s_f.transform(x_test)
model = MLPClassifier(hidden_layer_sizes=(32), activation="relu", random_state=1, max_iter=100)
result = model.fit(x_train, y_train)
l_pred = result.predict(x_test)
print(l_pred)
print(result.score(x_test, y_test))
fig = plot_confusion_matrix(result, x_test, y_test, display_labels=target_names)
fig.figure_.suptitle("Confusion Matrix for Iris Dataset")
plt.show()
print(classification_report(y_test, l_pred, target_names=target_names))


#-----------------------------Question 13-----------------------------------------------------------------------------------------------------------------------------------------

x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2,random_state=0)
print(x_train.shape)
print(y_train.shape)
s_f = StandardScaler()
x_train = s_f.fit_transform(x_train)
x_test = s_f.transform(x_test)


#learning_rate_init=0.002
model = MLPClassifier(hidden_layer_sizes=(32), activation="relu", random_state=1, max_iter=100,learning_rate_init=0.002)
result = model.fit(x_train, y_train)
l_pred = result.predict(x_test)
print(l_pred)
print(result.score(x_test, y_test))
fig = plot_confusion_matrix(result, x_test, y_test, display_labels=target_names)
fig.figure_.suptitle("Confusion Matrix for Iris Dataset")
plt.show()
print(classification_report(y_test, l_pred, target_names=target_names))


#learning_rate_init=0.5
model = MLPClassifier(hidden_layer_sizes=(32), activation="relu", random_state=1, max_iter=100, learning_rate_init=0.5)
result = model.fit(x_train, y_train)
l_pred = result.predict(x_test)
print(l_pred)
print(result.score(x_test, y_test))
fig = plot_confusion_matrix(result, x_test, y_test, display_labels=target_names)
fig.figure_.suptitle("Confusion Matrix for Iris Dataset")
plt.show()
print(classification_report(y_test, l_pred, target_names=target_names))


#learning_rate_init=1
model = MLPClassifier(hidden_layer_sizes=(32), activation="relu", random_state=1, max_iter=100, learning_rate_init=1)
result = model.fit(x_train, y_train)
l_pred = result.predict(x_test)
print(l_pred)
print(result.score(x_test, y_test))
fig = plot_confusion_matrix(result, x_test, y_test, display_labels=target_names)
fig.figure_.suptitle("Confusion Matrix for Iris Dataset")
plt.show()
print(classification_report(y_test, l_pred, target_names=target_names))
