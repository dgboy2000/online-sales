from LearnerBase import LearnerBase
import numpy as np
from scipy import linalg

class LinearRegression(object):
  def __init__(self, debug=False):
    self.debug = debug
    self.params = None
    
  def cross_validate(self, dataset, num_folds):
    pass
    
  def train(self, features, labels):
    num_samples, num_features = features.shape
    if self.debug:
      print "Running linear regression with %d features and %d observations..." %(num_features, num_samples)
    A = np.hstack((features, np.ones((num_samples,1))))
    self.params, residues, rank, s = linalg.lstsq(A, labels)
    
  def predict(self, features):
    num_samples, num_features = features.shape
    A = np.hstack((features, np.ones((num_samples,1))))
    probs = A.dot(self.params)
    return np.minimum(np.maximum(probs, 0.1*np.ones(num_samples)), 0.9*np.ones(num_samples))

    
LearnerBase.register(LinearRegression)
    
    
    
    
    
    
    
    
    
    
    
    