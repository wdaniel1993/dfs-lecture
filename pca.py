# Principal Component Analysis (PCA) Demo
# Inspired froms: http://sebastianraschka.com/Articles/2015_pca_in_3_steps.html#a-summary-of-the-pca-approach

import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

class iris_pca:
    def load_iris_data(self, path):
        # load the iris data (https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data)
        df = pd.read_csv(
            filepath_or_buffer=path,
            header=None,
            sep=',')

        df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
        df.dropna(how="all", inplace=True) # drops the empty line at file-end

        # split the data table into feature data x and class labels y
        x = df.ix[:,0:4].values # the first 4 columns are the features
        y = df.ix[:,4].values   # the last column is the class label
        return x, y
    
    def plot_scatter_matrix(self):
        df = pd.read_csv('iris.data')
        df.columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Species']
        sns.set(font_scale=2.5)
        ax = sns.pairplot(df, hue="Species", size=4, aspect=1.3)
        ax.savefig('iris_scatter.pdf')

    def plot_feature_histograms(self):
        print(self.features)
        # easier to work with the labels
        with plt.style.context('seaborn-whitegrid'):
            plt.figure(figsize=(8, 6))
            for cnt in range(4):
                plt.subplot(2, 2, cnt+1)
                for lab in (self.labels):
                    plt.hist(self.x[self.y==lab, cnt],
                             label=lab,
                             bins=10,
                             alpha=0.5,)
                    plt.xlabel(self.features[cnt])
            plt.legend(loc='upper right', fancybox=True, fontsize=8)

            plt.tight_layout()
            plt.show()
            
    def plot_principal_components_barplot(self, eig_vals):
        sns.set_style('whitegrid')
        ax = sns.barplot(x=['PC1', 'PC2', 'PC3', 'PC4'], y=eig_vals / sum(eig_vals) * 100)
        ax.set_ylim(0,100)
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., height + 3, '{0:.2f}\%'.format(height), ha="center", size=15)
        #ax.set(xlabel='Principal components', ylabel='Explained variance (%)')
        ax.set_xlabel('Principal components', fontsize=20)
        ax.set_ylabel('Explained variance (\%)', fontsize=20)
        ax.tick_params(labelsize=15)
        plt.savefig('iris_principal_components.pdf')

    def __init__(self):
        self.labels = ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica')
        self.features = ('sepal length [cm]', 'sepal width [cm]', 'petal length [cm]', 'petal width [cm]')

        self.x, self.y = self.load_iris_data('./iris.data')
        # plot histograms of class labels for each feature in the data 
        # self.plot_feature_histograms()
        #self.plot_scatter_matrix()

        # Since PCA yields a feature subspace that maximizes the variance along the axes, 
        # it makes sense to standardize the data, especially if it was measured on different scales
        x_std = StandardScaler().fit_transform(self.x)
        #print('Normalized feature data \n%s' %x_std)
        
        # calculate the covariance matrix
        # verbose way (for demonstration):
        # calculate the sample mean
        # mean_vec = np.mean(x_std, axis=0)
        # print('Mean vector \n%s' %mean_vec)
        # cov_mat = (x_std - mean_vec).T.dot((x_std - mean_vec)) / (x_std.shape[0] - 1)
        
        # compact way:
        cov_mat = np.cov(x_std.T)
        print('Covariance matrix \n%s' %cov_mat)

        # perform an eigen decomposition of the covariance matrix to find the eigen vectors
        # which give the direction of the principal components
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)
        total = sum(eig_vals)
        print('total: ', total)
        print('eigenvectors: ', eig_vecs)
        print('eigenvalues: ', eig_vals)
        
        
        w = np.hstack((eig_vecs[:,0].reshape(4,1), eig_vecs[:,1].reshape(4,1)))
        print('W: ', w)
        y = x_std.dot(w)
        #print('Y: ', y)
        
#        df = pd.DataFrame(y)
#        df.columns = [ 'Principal component 1', 'Principal component 2' ]
#        df['Species'] = self.y
#        print(df)
#        lm = sns.lmplot('Principal component 1', 'Principal component 2', data=df, hue='Species', size=4, aspect=1.3, fit_reg=False)
#        ax = lm.axes
#        ax[0,0].set_xlim(-4, 4)
#        ax[0,0].set_ylim(-4, 4)
#        plt.savefig('iris_principal_components_scatter.pdf')
        c = np.zeros((4, 2))
        
        print(len(y[:,0]))
        print('c: ', c)
        for i in range(4):
            for j in range(2):
                r, s = pearsonr(x_std[:,i], y[:,j])
                print('r: ', r * r)
                c[i, j] = r * r

        print('contributions: {0}'.format(c))

# create an iris_pca instance
iris = iris_pca()
