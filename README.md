# Naive Bayes Classifier for text classification #  
  This repository comprises of Naive Bayes classifier with text parser for generating(fitting and vectorizing) feature vectors from text
  
## ParseText ##  
  Vectorize textfile and returns a sparse matrix. 
  
  Methods:
	
    ParseText.ParseText.featureidx(self)  
		return mapping from feature/token/word to index
		  
	ParseText.ParseText.featureidx(self)  
		return the list of words used as features  
	  
	ParseText.ParseText.getfilenames(self)  
		return the names of the labels used
  
	ParseText.ParseText.getlabels(self)  
		return the labels of the fit data  
	
	ParseText.ParseText.fit_transform(self, filename, delimiter=',')  
		Read csv of the format <samplename> <label> <attribute> <frequency> ...)  
		and return the feature vectors and the labels
  
	ParseText.ParseText.vectorize(self, filename, delimiter=',')  
		Vectorize a file given the features created using fit  
	  
	  
## NaiveBayes ##  
	Class for running Naive Bayes on input vectors  
  
	Methods:  
	__init__(self, smoothing=1.0, tfidf=False):  
		set the smoothing/alpha param for smoothing the likelihood, default laplacian smoothing with smoothing = 1.0  
		Set tfidf to True to use tfidf weights instead of count, idf = tf*(log((n+1)/(wordindocs+1))+1)
  
	NaiveBayes.NaiveBayes.fit(self, X, y)  
		return fitted likelihood and prior probabilities.  
		X should be a list, ndarray or Sparse matrix
		y shoudl be a list or ndarray 
		Note: Both Likelihood and prior are log values, inorder to get the actual need to apply exp()
	  
	NaiveBayes.NaiveBayes.predict(self, X)
		return the predictions for the input X. X should be a list, np array or Sparse matrix
		
	 

    
