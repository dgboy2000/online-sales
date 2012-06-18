from LearnerBase import LearnerBase
import numpy as np
from scipy import linalg

class LinearRegression(object):
  def __init__(self, debug=False):
    self.debug = debug
    self.params = None
    
  def cross_validate(self, dataset, num_folds):
    pass
    
  def train(self, dataset):
    num_features = dataset.getNumFeatures()
    params = np.zeros((num_features+1, 12))
    if self.debug:
      print "Running linear regression with %d features..." %(num_features)
    
    for month_ind in range(12):
      month_features = dataset.getFeaturesForMonth(month_ind)
      num_samples = month_features.shape[0]
      
      if self.debug:
        print "Learning on month %d of 12 with %d samples..." %(month_ind+1, num_samples)
      
      A = np.hstack((month_features, np.ones((num_samples,1))))
      
      month_params, residues, rank, s = linalg.lstsq(A, dataset.getSalesForMonth(month_ind))
      params[:, month_ind] = month_params
    
    self.params = params
    
  def predict(self, dataset):
    features = dataset.getFeatures()
    
    num_samples, num_features = features.shape
    A = np.hstack((features, np.ones((num_samples,1))))
    sales = A.dot(self.params)
    return np.maximum(sales, np.zeros(sales.shape))

    
LearnerBase.register(LinearRegression)
    
    
    
    
    
    
    
    
    
    
    
    