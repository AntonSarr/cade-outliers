import itertools

import numpy as np
from sklearn.model_selection import train_test_split


class CADEOutliers():
    """Class for outliers detection with CADE (Classifier Adjusted Density Estimation)"""
    
    _supported_A_dist = ["uniform", "grid"]
    
    def __init__(self, classifier, A_dist, A_size):
        """Create CADE outlier detection object.
        
        Parameters
        ----------
        classifier : model
            sklearn classifier model supporting fit and predict_proba methods.
        A_dist : string
            Distribution for generating artificial anomalies. Can be "uniform" or "grid".
        A_size : float | int
            If float, then the number of artifical anomalies is |X| * A_size, where X - analyzed data. 
            If int, then the number of artifical anomalies is just A_size.
        """
        # Set classifier
        self.classifier = classifier
        
        # Set the distribution kind
        if A_dist not in self._supported_A_dist:
            raise ValueError(f"{A_dist} distribution is not supported yet for generating artifical anomalies. Choose from the following list: {self._supported_A_dist}")
        self.A_dist = A_dist
        
        # Set the number of artifical anomalies to generate
        if isinstance(A_size, int) and A_size >= 1:
            self.sample_size = A_size
        elif isinstance(A_size, float) and A_size >= 0:
            self.sample_size = None
        else:
            raise ValueError("A_size must be either int[1, inf) of float[0, inf)")
        self.A_size = A_size
        
    def _generate_A(self, dist, attrs):
        """Generate artifical anomalies.
        
        Parameters
        ----------
        dist : string
            Distribution for generating artificial anomalies. Can be "uniform" or "grid".
        attrs : dict
            Key-value arguments for generating artifical anomalies. 
            If dist is "uniform", attrs = {'low': array, 'high': array, 'size': (sample_size, n_dim)} with min and high thresholds for each dimension.
            If dist is "grid", attrs = {'low': array, 'high': array, 'num': array} with values for each dimension.
        """
        if dist == 'uniform':
            generator = np.random.uniform
        elif dist == 'grid':
            generator = lambda low, high, num: np.array([list(x) for x in itertools.product(*[np.linspace(start, stop, num) for start, stop, num in zip(low, high, num)])])
        else:
            raise ValueError(f"{dist} distribution is not supported yet for generating artifical anomalies. Choose from the following list: {self._supported_A_dist}")
        
        A = generator(**attrs)
        
        return A
        
    def outliers_ranking(self, X):
        """Return an array for ranking outliers in X.
        
        Parameters
        ----------
        X : np.ndarray
            Array with samples
        """
        if self.sample_size is None:
            sample_size = int(len(X) * self.A_size) + 1
        else:
            sample_size = int(self.sample_size)
        
        if self.A_dist == 'uniform':
            n_dim = X.shape[1]
            attrs = {
                'low': X.min(axis=0),
                'high': X.max(axis=0),
                'size': (sample_size, n_dim)
            }
            # calculate uniform probability density
            P_A = 1
            for s in X.max(axis=0) - X.min(axis=0):
                P_A /= s
            P_A = np.array([P_A] * X.shape[0])
        elif self.A_dist == 'grid':
            n_dim = X.shape[1]
            attrs = {
                'low': X.min(axis=0),
                'high': X.max(axis=0),
                'num': [int(pow(sample_size, 1 / n_dim)) + 1 for i in range(n_dim)]
            }
            # calculate uniform probability density
            P_A = 1 
            for s in X.max(axis=0) - X.min(axis=0):
                P_A /= s
            P_A = np.array([P_A] * X.shape[0])
        else:
            attrs = None
        
        A = self._generate_A(self.A_dist, attrs)
        
        combined_data = np.vstack([A, X])
        target = np.hstack([np.zeros(X.shape[0]), np.ones(A.shape[0])])
        
        X_train, X_test, y_train, y_test = train_test_split(combined_data, target, test_size=0.33)
        self.classifier.fit(X_train, y_train)
        
        predictions = self.classifier.predict_proba(X)[:, 0]
        
        anomaly_score = P_A * (A.shape[0] / X.shape[0]) * (predictions / (1 - predictions + 1e-5))
        
        return anomaly_score