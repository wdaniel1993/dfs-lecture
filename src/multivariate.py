from math import atan
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde 
from sklearn.preprocessing import StandardScaler
from matplotlib import rc
import matplotlib.pyplot as plt
import numpy as np

plt.set_cmap('jet')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['font.size'] = 10
plt.rcParams['text.latex.preamble'] = r'\usepackage{libertine},\usepackage[libertine]{newtxmath},\usepackage{amsmath}'

# define a multivariate distribution (2d)
cov = [[0.4, 0.2], [0.2, 0.2]]
center = [0, 0]
rv = multivariate_normal(center , cov )

# sample the distribution
s = rv.rvs(3000)
x, y = s[:,0], s[:,1]
# standardize the samples
s_std = StandardScaler().fit_transform(s)
s_mean = np.mean(s_std)
# calculate the covariance matrix of the samples
s_cov = np.cov(s.T)
xy = s.T
z = gaussian_kde(xy)(xy)
# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
xx, yy, z = x[idx], y[idx], z[idx]
plt.scatter(xx, yy, c=z, s=10, alpha=0.3, edgecolor='')

# draw eigen vectors
eigval, eigvec = np.linalg.eig(s_cov)
print('eigval: ', eigval)
print('eigvec: ', eigvec)

scale = 2
ax, ay = np.mean(x), np.mean(y)
for i in range(len(eigval)):
    v = eigvec[:,i]
    
    xlen = v[0] * eigval[i] * scale
    ylen = v[1] * eigval[i] * scale
    # plt.plot([np.mean(x), v[0]], [np.mean(y), v[1]], linewidth=1, c='black')
    plt.arrow(ax, ay, xlen, ylen, width=0.01)
    plt.text(ax + xlen + 0.01, ay + ylen + 0.05, '\huge $v_{:d}$'.format(i+1))

plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.savefig('2d_distribution.pdf')
plt.show()

