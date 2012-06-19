from LearnerBase import LearnerBase
import numpy as np
from scipy import linalg
import Score

class RidgeRegression(object):
  # k_values = [8, 9, 10, 11, 12.5, 14]
  k_values = [6, 7]
  def __init__(self, k=10, debug=False):
    self.debug = debug
    self.k_list = None
    self.params = None
    
  def _train(self, dataset, k, k_list = None):
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

      if k_list:
        k = k_list[month_ind]
      
      month_params = np.linalg.inv(A_T.dot(A) + k**2 * np.identity(num_features+1)).dot(A_T).dot(dataset.getSalesForMonth(month_ind))
      params[:, month_ind] = month_params
    
    self.params = params
    
  def cross_validate(self, dataset, num_folds):
    dataset.createFolds(num_folds)

    best_rmsle_list = [float("inf")] * 12
    self.k_list = [0] * 12

    for k in RidgeRegression.k_values:
      if self.debug:  
        print "Running ridge regression with k=%f on %d folds" %(k, num_folds)

      score = Score.Score()

      for fold_ind in range(num_folds):
        fold_train = dataset.getTrainFold(fold_ind)
        fold_test = dataset.getTestFold(fold_ind)
        self._train(fold_train, k)
        score.addFold(fold_test.getSales(), self.predict(fold_test))

      for month_ind in range(12):
        cur_rmsle = score.getRMSLE(month_ind)

        if cur_rmsle < best_rmsle_list[month_ind]:
          best_rmsle_list[month_ind] = cur_rmsle
          self.k_list[month_ind] = k
        
  def train(self, dataset):
    if self.debug:
      print "Training ridge regression with k_list: %s" %(self.k_list)

    self._train(dataset, None, self.k_list)
    
  def predict(self, dataset):
    features = dataset.getFeatures()
    
    num_samples, num_features = features.shape
    A = np.hstack((features, np.ones((num_samples,1))))
    sales = A.dot(self.params)
    return np.maximum(sales, np.zeros(sales.shape))

    
LearnerBase.register(RidgeRegression)
