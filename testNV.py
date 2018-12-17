from NaiveBayes import NaiveBayes
from ParseText import ParseText
import numpy as np
import timeit


def calldef():
    X = np.random.randint(5, size=(6, 100))
    y = np.array([1, 2, 3, 4, 5, 6])
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(X, y)
    clf.predict(X)

def callcustom():
    X = np.random.randint(5, size=(6, 100))
    y = np.array([1, 2, 3, 4, 5, 6])
    clf = NaiveBayes()
    clf.fit(X, y)
    clf.predict(X)

def parsecall():
    parser = ParseText()
    parser.fit(r'./train', delimiter=' ')


if __name__ == '__main__':
    # print(timeit.timeit(calldef, number=500))
    # print(timeit.timeit(callcustom, number=5000))
    # print(timeit.timeit(parsecall, number=1))
    parser = ParseText()
    X, y = parser.fit(r'./train', delimiter=' ')
    clf = NaiveBayes()
    clf.fit(X, y)

    X, ya = parser.vectorize(r'./test', delimiter=' ')
    yp = clf.predict(X)
    print(ya, yp)
    # from sklearn.metrics import confusion_matrix, accuracy_score
    # print(confusion_matrix(ya, yp), accuracy_score(ya, yp))


