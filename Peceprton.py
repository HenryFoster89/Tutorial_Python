# Implementing a perceptron learning algorithm in Python

# Perceptron Object => Can learn data from a Fit Method, Predict method makes prediction

import numpy as np

class Perceptron(object):
    """Perceptron classifier.

    Parameters
    -----------
    eta : float
        Learning Rate (between 0.0 and 1.0)
    n_iter: int
        Passes over the training dataset.
    random_state: int
        Random number generator seed for random weight initialization.

    Attributes
    ----------
    w_ : 1d-array
        Weights after fitting.
    error_: list
        Number of missclassification (updates) in each epoch.
    """

    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------          
        X:  {array-like}, shape = [n_examples, n_features]
            Training vectors, when n_examples is the number of examples and n_features is the number of features.
        y:  array-like, shape = [n_examples]
            Target values.

        Returns
        -------
        self: object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1]) 

