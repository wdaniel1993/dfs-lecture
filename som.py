from sklearn.datasets import load_iris
from mvpa2.suite import SimpleSOMMapper
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

df = pd.read_csv(filepath_or_buffer='iris.data', header=None, sep=',')
#df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

# split the data table into feature data x and class labels y
x = df.ix[:,0:3].values # the first 4 columns are the features
#y = df.ix[:,4].values   # the last column is the class label
#t = np.zeros(len(y), dtype=int)
#t[y == 'Iris-setosa'] = 0
#t[y == 'Iris-versicolor'] = 1
#t[y == 'Iris-virginica'] = 2

dim = 50
som = SimpleSOMMapper((dim, dim), 400, learning_rate=0.01)
x_std = StandardScaler().fit_transform(x)
som.train(x_std)

k = som.K

m = k[:,:,0]
im = plt.imshow(m, origin = 'lower', cmap='jet')
plt.colorbar(im, orientation='vertical')
frame = plt.gca()
frame.axes.get_xaxis().set_visible(False)
frame.axes.get_yaxis().set_visible(False)
#plt.xlim(0, dim-1)
#plt.ylim(0, dim-1)
#plt.savefig('colors_som_map_{:d}.pdf'.format(dim), bbox_inches='tight')
#
#mapped = som(x_std)
#markers = ['o','s','D']
#for i, m in enumerate(mapped):
#    plt.text(m[1], m[0], t[i], ha='center', va='center')


#colors = ['r','g','b']
#for i, m in enumerate(mapped):
#    plt.plot(m[0], m[1], markers[t[i]], markerfacecolor='', markeredgecolor=colors[t[i]],
#             markersize=12, markeredgewidth=2)
#    
#plt.axis([0,som.K.shape[0],0,som.K.shape[1]])
plt.show() # show the figure