import sklearn
import sklearn.cluster as clusters
import sklearn.datasets as datasets
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class Clustering: 

    def load_data(self):
        self.dataset = datasets.load_iris()
        self.X = self.dataset.data
        self.y = self.dataset.target
        return 'data loaded'

    def fit_kmeans(self):
        estimators = [('k_means_iris_8', clusters.KMeans(n_clusters=8)),
              ('k_means_iris_3', clusters.KMeans(n_clusters=3)),
              ('k_means_iris_bad_init', clusters.KMeans(n_clusters=3, n_init=1,
                                               init='random'))]
        
        fignum = 1
        titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']
        for name, est in estimators:
            fig = plt.figure(fignum, figsize=(4, 3))
            ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
            est.fit(self.X)
            labels = est.labels_

            ax.scatter(self.X[:, 3], self.X[:, 0], self.X[:, 2],
                    c=labels.astype(np.float), edgecolor='k')

            ax.w_xaxis.set_ticklabels([])
            ax.w_yaxis.set_ticklabels([])
            ax.w_zaxis.set_ticklabels([])
            ax.set_xlabel('Petal width')
            ax.set_ylabel('Sepal length')
            ax.set_zlabel('Petal length')
            ax.set_title(titles[fignum - 1])
            ax.dist = 12
            fignum = fignum + 1

        # Plot the ground truth
        fig = plt.figure(fignum, figsize=(4, 3))
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

        for name, label in [('Setosa', 0),
                            ('Versicolour', 1),
                            ('Virginica', 2)]:
            ax.text3D(self.X[self.y == label, 3].mean(),
                    self.X[self.y == label, 0].mean(),
                    self.X[self.y == label, 2].mean() + 2, name,
                    horizontalalignment='center',
                    bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
        # Reorder the labels to have colors matching the cluster results
        y = np.choose(self.y, [1, 2, 0]).astype(np.float)
        ax.scatter(self.X[:, 3], self.X[:, 0], self.X[:, 2], c=self.y, edgecolor='k')

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('Petal width')
        ax.set_ylabel('Sepal length')
        ax.set_zlabel('Petal length')
        ax.set_title('Ground Truth')
        ax.dist = 12

        fig.show()

        return fig