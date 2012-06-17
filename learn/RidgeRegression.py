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
    
  def _train_with_k(self, features, labels, k):
    # http://en.wikipedia.org/wiki/Tikhonov_regularization
    # <math>\hat{x} = (A^{T}A+ \Gamma^{T} \Gamma )^{-1}A^{T}\mathbf{b}</math>
    num_samples, num_features = features.shape
    A = np.hstack((features, np.ones((num_samples,1))))
    A_T = A.transpose()
    self.params = np.linalg.inv(A_T.dot(A) + k**2 * np.identity(num_features+1)).dot(A_T).dot(labels)
    
  def cross_validate(self, dataset, num_folds):
    dataset.createFolds(num_folds)
    best_k = None
    best_score = float("inf")
    for k in RidgeRegression.k_values:
      if self.debug:  
        print "Running ridge regression with k=%f on %d folds" %(k, num_folds)
      cur_score = 0
      for fold_ind in range(num_folds):
        fold_train = dataset.getTrainFold(fold_ind)
        fold_test = dataset.getTestFold(fold_ind)
        self._train_with_k(fold_train.getFeatures(), fold_train.getLabels(), k)
        fold_score = Score.Score(fold_test.getLabels(), self.predict(fold_test.getFeatures()))
        cur_score += fold_score.getLogLoss()
      cur_score /= num_folds
      if cur_score < best_score:
        if self.debug:
          print "Achieved new best score %f" %cur_score
        best_k = k
        best_score = cur_score
        
    self.k = best_k
        
  def train(self, features, labels):
    self._train_with_k(features, labels, self.k)
    
  def predict(self, features):
    num_samples, num_features = features.shape
    A = np.hstack((features, np.ones((num_samples,1))))
    probs = A.dot(self.params)
    return np.minimum(np.maximum(probs, 0.01*np.ones(num_samples)), 0.99*np.ones(num_samples))

    
LearnerBase.register(RidgeRegression)
    
    
    
    
    
    
    
    
    
    
    
