from LearnerBase import LearnerBase
import numpy as np
from scipy import linalg

class LinearRegression(object):
  def __init__(self, debug=False):
    self.debug = debug
    self.params = None
    
  def cross_validate(self, dataset, num_folds):
    pass
    
  def train(self, features, sales):    
    num_samples, num_features = features.shape
    params = np.zeros((num_features+1, 12))
    if self.debug:
      print "Running linear regression with %d features and %d observations..." %(num_features, num_samples)
    A = np.hstack((features, np.ones((num_samples,1))))
    
    for month_ind in range(12):
      if self.debug:
        print "Learning on month %d of 12" %(month_ind+1)
      month_params, residues, rank, s = linalg.lstsq(A, sales[:, month_ind])
      params[:, month_ind] = month_params
    
    self.params = params
    
  def predict(self, features):
    num_samples, num_features = features.shape
    A = np.hstack((features, np.ones((num_samples,1))))
    sales = A.dot(self.params)
    return np.maximum(sales, np.zeros(sales.shape))

    
LearnerBase.register(LinearRegression)
    
    
    
    
    
    
    
    
    
    
    
    