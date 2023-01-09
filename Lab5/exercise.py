import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler

#-----------------------------Question 1-----------------------------------------------------------------------------------------------------------------------------------------

data_read = pd.read_csv("diabetes.csv")
features_appropriate = data_read.drop('Outcome', 1)
features_appropriate.head()

#-----------------------------Question 2-----------------------------------------------------------------------------------------------------------------------------------------

scan = MinMaxScaler()
features = scan.fit_transform(features_appropriate)
print(features.shape)
features[:1][:]

#-----------------------------Question 3-----------------------------------------------------------------------------------------------------------------------------------------

diabetes = 10
not_diabetes = 10
som = MiniSom(diabetes, not_diabetes, features.shape[1], neighborhood_function='gaussian', sigma=1.0,learning_rate=0.7, random_seed=5)

som.pca_weights_init(features)
som.train_batch(features, 500, verbose=True)

#-----------------------------Question 6-----------------------------------------------------------------------------------------------------------------------------------------

output = som.labels_map(features, data_read.Outcome)
legend1 = {'Do not have diabetes(0)': 'black', 'Has diabetes(1)': 'red'}
graph1_color = {0: 'black', 1: 'red'}

plt.figure(figsize=(diabetes, not_diabetes))
for a, b in output.items():
    b = list(b)
    x = a[0] + 0.4
    y = a[1] - 0.3
    for i, c in enumerate(b):
        off_set = (i + 2) / 2 - 0.5
        plt.text(x, y + off_set, c, color=graph1_color[c], fontsize=10)
plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=0.4)
plt.xticks(np.arange(diabetes + 1))
plt.yticks(np.arange(not_diabetes + 1))
plt.grid()

legend_elements = [Patch(facecolor=clr,
                         edgecolor='w',
                         label=l) for l, clr in legend1.items()]
plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, .95))

plt.show()

plt.figure(figsize=(diabetes, not_diabetes))

plt.pcolor(som.distance_map().T, cmap='bone_r')  
plt.colorbar()

target = data_read.Outcome


markers = ['D', 'o']
colors = ['C10', 'C3']
for j, k in enumerate(features):
	# getting the winner
    w = som.winner(k)
    plt.plot(w[0] + 0.5, w[1] + 0.5, markers[target[j] - 1], markerfacecolor='None',
             markeredgecolor=colors[target[j]], markersize=10, markeredgewidth=2)

plt.show()

#-----------------------------Question 7-----------------------------------------------------------------------------------------------------------------------------------------

plt.figure(figsize=(diabetes, not_diabetes))
frequencies = som.activation_response(features)
plt.pcolor(frequencies.T, cmap='Reds')
plt.colorbar()
plt.show()

#-----------------------------Question 8-----------------------------------------------------------------------------------------------------------------------------------------

l_names = {0: 'Do not have diabetes', 1: 'Has diabetes'}

w_x, w_y = zip(*[som.winner(d) for d in features])
w_x = np.array(w_x)
w_y = np.array(w_y)



plt.figure(figsize=(diabetes, not_diabetes))
plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=.4)
plt.colorbar()

for c in np.unique(target):
    idx_target = target == c
    plt.scatter(w_x[idx_target] + .5 + (np.random.rand(np.sum(idx_target)) - .5) * .8,
                w_y[idx_target] + .5 + (np.random.rand(np.sum(idx_target)) - .5) * .8,
                s=50, c=colors[c], label=l_names[c])
plt.legend(loc='upper right')
plt.grid()
plt.show()

import matplotlib.gridspec as gridspec

labels_map = som.labels_map(features, [l_names[t] for t in target])

fig = plt.figure(figsize=(diabetes, not_diabetes))
the_grid = gridspec.GridSpec(diabetes, not_diabetes, fig)
for position in labels_map.keys():
    label_fracs = [labels_map[position][l] for l in l_names.values()]
    plt.subplot(the_grid[diabetes - 1 - position[1],
                         position[0]], aspect=1)
    patches, texts = plt.pie(label_fracs)

plt.legend(patches, l_names.values(), bbox_to_anchor=(-5, 5), ncol=6)
plt.show()

som = MiniSom(diabetes, not_diabetes, features.shape[1], neighborhood_function='gaussian', sigma=1.0, learning_rate=0.7,random_seed=5)

max_iter = 100
q_error = []
t_error = []

for i in range(max_iter):
    rand_i = np.random.randint(len(data_read))
    som.update(features[rand_i], som.winner(features[rand_i]), i, max_iter)
    q_error.append(som.quantization_error(features))
    t_error.append(som.topographic_error(features))

plt.plot(np.arange(max_iter), q_error, label='quantization error')
plt.plot(np.arange(max_iter), t_error, label='topographic error')
plt.ylabel('quantization error')
plt.xlabel('iteration index')
plt.legend()
plt.show()