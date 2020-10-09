import sklearn
import sklearn.cluster as clusters
import sklearn.datasets as datasets


class clustering: 
    # def __init__(self): 
    #     self.X, self.y = 0
    #     self.clustering_algorithm = None
    
    def fit():
        self.dataset = datasets.load_iris(return_X_y=True)
        X, y = load_boston(return_X_y=True)
        self.clusterng_algorithm = clusters.DBSCAN().fit(X)
        return type(self.clustering_algorithm)