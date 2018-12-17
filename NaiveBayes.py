import numpy as np
import scipy

class NaiveBayes:
    def __init__(self, smoothing=1.0):
        self.numclasses = 0
        self.priors = None
        self.smoothing = smoothing
        self.numsamples = 0.0
        self.likelihood = None
        self.numattributes = 0.0
        self.labels = None

    def fit(self, X, y):
        assert type(y) in[np.ndarray, list], 'expecting array or numpy array'
        assert type(X) in [np.ndarray, list, scipy.sparse.bsr.bsr_matrix], 'expecting array or numpy array'

        if type(X) == scipy.sparse.bsr.bsr_matrix:
            y = np.asarray(y)
            X = X.toarray()

        elif type(X) == list:
            y = np.asarray(y)
            X = np.toarray(X)

        self.labels = np.unique(y)
        self.numclasses = self.labels.shape[0]
        self.numattributes = X.shape[1]
        self.priors = np.zeros(self.numclasses, dtype=float)
        self.numsamples = y.shape[0]
        self.likelihood = np.zeros((self.numclasses, X.shape[1]), dtype=float)
        updcount = float(self.smoothing*self.numattributes)

        self.priors = np.log(np.asarray([np.where(y == cl)[0].shape[0]/float(self.numsamples)
                                         for cl in self.labels]))

        def attlikelihood(X):
            attsum = np.sum(X)
            return np.log((X + self.smoothing)/(updcount + attsum))

        labelwordsum = np.squeeze(np.asarray([np.sum(X[np.where(y == cl), :], axis=1)
                                                 for cl in self.labels]), axis=1)

        self.likelihood = np.apply_along_axis(attlikelihood, 1, labelwordsum)
        # self.likelihood = np.squeeze(np.asarray([np.sum(self.likelihood[np.where(y == cl), :], axis=1)
        #                                          for cl in self.labels]), axis=1)
        return self.likelihood, self.priors

    def predict(self, X):
        assert type(X) in [np.ndarray, list, scipy.sparse.bsr.bsr_matrix], 'expecting array or numpy array'

        if type(X) == scipy.sparse.bsr.bsr_matrix:
            X = X.toarray()

        elif type(X) == list:
            X = np.asarray(X)


        # print(self.priors)
        # print(self.likelihood)
        # print (self.labels)

        def predictsample(row):
            return self.labels[np.argmax(np.dot(self.likelihood, row) + self.priors, axis=0)]

        return np.apply_along_axis(predictsample, 1, X)
