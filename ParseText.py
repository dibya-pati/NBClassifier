import numpy as np
from scipy.sparse import bsr_matrix
import operator

class ParseText:
    def __init__(self):
        self.features = None
        self.samplename = None
        self.labels = None
        self.featurevector = None
        self.numsamples = 0
        self.fpos = None
        self.labelnames = None

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

    '''Read csv of the format 
    <samplename> <label> <attribute> <frequency> ...
    '''
    def fit(self, filename, delimiter=','):
        featureset = set()
        namepos = 0
        labelpos = 1
        
        # read from file, use an unused separator to get it as a whole line
        alltext = np.genfromtxt(filename, dtype='str', delimiter='\UFFFD')
        
        # split to tokens
        tokens = np.char.split(alltext, delimiter)
        
        # extract the names of the samples, labels, number of samples used
        self.samplename = np.asarray(map(lambda x: x[namepos], tokens))
        labels = np.asarray(map(lambda x: x[labelpos], tokens))
        self.labelnames, self.labels = np.unique(labels, return_inverse=True)
        self.numsamples = self.labels.shape[0]

        # create a set of all words in the file
        map(lambda line:
            map(lambda valid: featureset.add(valid.lower()), filter(lambda word: word.isalpha(), line[2::2]))
            , tokens)

        # convert to np array
        self.features = np.asarray(sorted(list(featureset)))

        # create a map of word to position
        self.fpos = {k: v for v, k in enumerate(self.features)}

        # create a list of lists contains (words, freq, line num)
        tokendarray = map(lambda linezip:
                          filter(lambda wzip: wzip[0].lower() in self.fpos,
                                 (zip(linezip[0][2::2],
                                      linezip[0][3::2], [linezip[1]]*((len(linezip[0]) - 2) / 2)))),
                          zip(tokens, range(self.numsamples)))

        # flatten the list of lists
        tokendarray = reduce(operator.add, tokendarray)

        # remap the feature name to index
        tokendarray = map(lambda x: (self.fpos[x[0]], x[1], x[2]), tokendarray)

        # convert to nd array
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

        # split to tokens
        tokens = np.char.split(alltext, delimiter)

        # create a list of lists contains (words, freq, line num)
        tokendarray = map(lambda linezip:
                          filter(lambda wzip: wzip[0].lower() in self.fpos,
                                 (zip(linezip[0][2::2],
                                      linezip[0][3::2], [linezip[1]] * ((len(linezip[0]) - 2) / 2)))),
                          zip(tokens, range(self.numsamples)))

        # flatten the list of lists
        tokendarray = reduce(operator.add, tokendarray)

        # remap the feature name to index
        tokendarray = map(lambda x: (self.fpos[x[0]], x[1], x[2]), tokendarray)

        # convert to nd array
        tokendarray = np.asarray(tokendarray)

        data = tokendarray[:, 1].astype(int)
        row = tokendarray[:, 2].astype(int)
        col = tokendarray[:, 0].astype(int)
        labels = np.asarray(map(lambda x: x[labelpos], tokens))
        featurevector = bsr_matrix((data, (row, col)), shape=(len(labels), self.features.shape[0]))

        return featurevector, labels










