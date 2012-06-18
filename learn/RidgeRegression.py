from LearnerBase import LearnerBase
import numpy as np
from scipy import linalg
import Score

class RidgeRegression(object):
  # k_values = [8, 9, 10, 11, 12.5, 14]
  k_values = [10]
  def __init__(self, k=10, debug=False):
    self.A = None
    self.A_T = None
    self.debug = debug
    self.k = k
    self.params = None
    
  def _train_with_k(self, dataset, k):
    # http://en.wikipedia.org/wiki/Tikhonov_regularization
    # <math>\hat{x} = (A^{T}A+ \Gamma^{T} \Gamma )^{-1}A^{T}\mathbf{b}</math>

    num_features = dataset.getNumFeatures()
    params = np.zeros((num_features+1, 12))
    if self.debug:
      print "Running ridge regression with %d features..." %(num_features)
    
    for month_ind in range(12):
      month_features = dataset.getFeaturesForMonth(month_ind)
      num_samples = month_features.shape[0]
      
      if self.debug:
        print "Learning on month %d of 12 with %d samples..." %(month_ind+1, num_samples)
      
      A = np.hstack((month_features, np.ones((num_samples,1))))
      A_T = A.transpose()
      
      month_params = np.linalg.inv(A_T.dot(A) + k**2 * np.identity(num_features+1)).dot(A_T).dot(dataset.getSalesForMonth(month_ind))
      params[:, month_ind] = month_params
    
    self.params = params
    
  def cross_validate(self, dataset, num_folds):
    dataset.createFolds(num_folds)
    best_k = None
    best_rmsle = float("inf")
    for k in RidgeRegression.k_values:
      if self.debug:  
        print "Running ridge regression with k=%f on %d folds" %(k, num_folds)

      k_score = Score.Score()
      for fold_ind in range(num_folds):
        fold_train = dataset.getTrainFold(fold_ind)
        fold_test = dataset.getTestFold(fold_ind)
        self._train_with_k(fold_train, k)
        k_score.addFold(fold_test.getSales(), self.predict(fold_test))
        
      cur_rmsle = k_score.getRMSLE()
      if cur_rmsle < best_rmsle:
        if self.debug:
          print "Achieved new best rmsle %f" %cur_rmsle
        best_k = k
        best_rmsle = cur_rmsle
        
    self.k = best_k
        
  def train(self, dataset):
    self._train_with_k(dataset, self.k)
    
  def predict(self, dataset):
    features = dataset.getFeatures()
    
    num_samples, num_features = features.shape
    A = np.hstack((features, np.ones((num_samples,1))))
    sales = A.dot(self.params)
    return np.maximum(sales, np.zeros(sales.shape))

    
LearnerBase.register(RidgeRegression)
    
    
    
    
    
    
    
    
    
    
    
