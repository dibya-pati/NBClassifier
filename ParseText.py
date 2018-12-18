import numpy as np
from scipy.sparse import bsr_matrix

class ParseText:
    def __init__(self):
        self.features = None
        self.samplename = None
        self.labels = None
        self.featurevector = None
        self.numsamples = 0
        self.fpos = None
        self.labelnames = None
        self.labelmapper = None

    '''return mapping from feature(word) to index'''
    def featureidx(self):
        return self.fpos

    '''return the list of words used as features'''
    def getfeaturenames(self):
        assert self.features is not None, "Use fit to set the features first"
        return self.features

    '''return the names of the labels used'''
    def getfilenames(self):
        assert self.samplename is not None, "Use fit to set the labels first"
        return self.samplename

    '''get the labels of the fit data'''
    def getlabels(self):
        assert self.labels is not None, "Use fit to set the labels first"
        return self.labels

    # '''flatten a nested list, faster than reduce ops'''
    # def _flatten(self, items, seqtypes=(list, tuple)):
    #     for i, x in enumerate(items):
    #         while i < len(items) and isinstance(items[i], seqtypes):
    #             items[i:i + 1] = items[i]
    #     return items

    '''Read csv of the format 
    <samplename> <label> <attribute> <frequency> ...
    '''
    def fit(self, filename, delimiter=','):
        featureset = set()
        namepos = 0
        labelpos = 1
        
        # read from file, use an unused separator to get it as a whole line
        alltext = np.genfromtxt(filename, dtype='str', delimiter='\UFFFD')
        
        # split to lines
        lines = np.char.split(alltext, delimiter)
        
        # extract the names of the samples, labels, number of samples used
        self.samplename = np.asarray([t[namepos] for t in lines])
        labels = np.asarray([t[labelpos] for t in lines])
        self.labelnames, self.labels = np.unique(labels, return_inverse=True)
        self.labelmapper = dict(zip(self.labelnames, range(self.labelnames.shape[0])))
        self.numsamples = self.labels.shape[0]

        # create a set of all words in the file
        for line in lines:
            words = line[2::2]
            for word in words:
                if word.isalpha():
                    featureset.add(word.lower())

        # convert to np array
        self.features = np.asarray(sorted(list(featureset)))

        # create a map of word to position
        self.fpos = {k: v for v, k in enumerate(self.features)}

        # create a list of lists contains (words, freq, line num)
        tokendarray = []
        for row, token in enumerate(lines):
            words = token[2::2]
            freq = token[3::2]
            for w, f in zip(words, freq):
                if w.lower() in self.fpos:
                    tokendarray.append((self.fpos[w], f, row))

        tokendarray = np.asarray(tokendarray)

        data = tokendarray[:, 1].astype(int)
        row = tokendarray[:, 2].astype(int)
        col = tokendarray[:, 0].astype(int)

        # convert to sparse matrix
        self.featurevector = bsr_matrix((data, (row, col)), shape=(self.numsamples, self.features.shape[0]))
        return self.featurevector, self.labels

    '''Vectorize a file given the features created using fit'''
    def vectorize(self, filename, delimiter=','):

        assert self.features is not None, "Use fit to set the features first"
        namepos = 0
        labelpos = 1

        # read from file, use an unused separator to get it as a whole line
        alltext = np.genfromtxt(filename, dtype='str', delimiter='\UFFFD')

        # split to lines
        lines = np.char.split(alltext, delimiter)

        # create tuples of word_index, freq, row
        tokendarray = []
        for row, token in enumerate(lines):
            words = token[2::2]
            freq = token[3::2]
            for w, f in zip(words, freq):
                if w.lower() in self.fpos:
                    tokendarray.append((self.fpos[w], f, row))

        # convert to nd array
        tokendarray = np.asarray(tokendarray)

        data = tokendarray[:, 1].astype(int)
        row = tokendarray[:, 2].astype(int)
        col = tokendarray[:, 0].astype(int)
        labels = np.asarray([self.labelmapper[t[labelpos]] for t in lines])
        featurevector = bsr_matrix((data, (row, col)), shape=(len(labels), self.features.shape[0]))

        return featurevector, labels










