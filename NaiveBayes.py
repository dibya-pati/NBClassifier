import numpy as np
from scipy.sparse import bsr_matrix


class NaiveBayes:
    def __init__(self, smoothing=1.0, tfidf=False):
        self.numclasses = 0
        self.priors = None
        self.smoothing = smoothing
        self.numsamples = 0.0
        self.likelihood = None
        self.numattributes = 0.0
        self.labels = None
        self.tfidf = tfidf


    def fit(self, X, y):
        assert type(y) in[np.ndarray, list], 'expecting array or numpy array'
        assert type(X) in [np.ndarray, list, bsr_matrix], 'expecting array or numpy array'

        if type(X) == bsr_matrix:
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

        # compute tfidf weights instead of the count
        if self.tfidf:
            alldoc = X.shape[0]
            # freqperdoc = np.sum(np.where(X > 0, 1, 0), axis=0)
            idf = np.log((alldoc + 1.0)/(np.sum(np.where(X > 0, 1, 0), axis=0) + 1.0)) + 1.0
            X = X.astype(float)/X.sum(axis=1)[:, None]
            X = X*idf

        def attlikelihood(X):
            attsum = np.sum(X)
            return np.log((X + self.smoothing)/(updcount + attsum))

        labelwordsum = np.squeeze(np.asarray([np.sum(X[np.where(y == cl), :], axis=1)
                                              for cl in self.labels]), axis=1)

        self.likelihood = np.apply_along_axis(attlikelihood, 1, labelwordsum)

        return self.likelihood, self.priors

    def predict(self, X):
        assert type(X) in [np.ndarray, list, bsr_matrix], 'expecting array or numpy array'

        if type(X) == bsr_matrix:
            X = X.toarray()

        elif type(X) == list:
            X = np.asarray(X)

        def predictsample(row):
            return self.labels[np.argmax(np.dot(self.likelihood, row) + self.priors, axis=0)]

        return np.apply_along_axis(predictsample, 1, X)
