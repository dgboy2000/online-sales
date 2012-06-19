from LearnerBase import LearnerBase
import numpy as np
from scipy import linalg
import Score

class RidgeRegression(object):
  num_iterations = 30
  min_k_list = np.array([.1] * 12)
  max_k_list = np.array([20.] * 12)
  
  def __init__(self, debug=False):
    self.debug = debug

    if debug:
      RidgeRegression.num_iterations = 1
      RidgeRegression.min_k_list = np.array([5.87542267, 4.674982, 8.13359445, 6.74467722, 6.08883212, 6.17201609, 5.29976006, 4.99435439, 5.72486321, 13.28838441, 5.22491289, 5.99526002])
      RidgeRegression.max_k_list = np.array([5.88439942, 4.68395875, 8.1425712, 6.75365397, 6.09780887, 6.18099284, 5.30873681, 5.00333114, 5.73383996, 13.29736116, 5.23388964, 6.00423677])

    self.k_list = None
    self.params = None
    self.best_rmsle_list = [float("inf")] * 12
    
  def _train(self, dataset, k, k_list = None):
    # http://en.wikipedia.org/wiki/Tikhonov_regularization
    # <math>\hat{x} = (A^{T}A+ \Gamma^{T} \Gamma )^{-1}A^{T}\mathbf{b}</math>

    num_features = dataset.getNumFeatures()
    params = np.zeros((num_features+1, 12))
    if 0 and self.debug:
      print "Running ridge regression with %d features..." %(num_features)
    
    for month_ind in range(12):
      month_features = dataset.getFeaturesForMonth(month_ind)
      num_samples = month_features.shape[0]
      
      if 0 and self.debug:
        print "Learning on month %d of 12 with %d samples..." %(month_ind+1, num_samples)
      
      A = np.hstack((month_features, np.ones((num_samples,1))))
      A_T = A.transpose()

      if k_list is not None:
        k = k_list[month_ind]
      
      month_params = np.linalg.inv(A_T.dot(A) + k**2 * np.identity(num_features+1)).dot(A_T).dot(dataset.getSalesForMonth(month_ind))
      params[:, month_ind] = month_params
    
    self.params = params

  def get_rmsle_list(self, dataset, num_folds, k_list):
    # print "get_rmsle_list k_list: %s" % k_list
    rmsle_list = []
    
    score = Score.Score()
    for fold_ind in range(num_folds):
      fold_train = dataset.getTrainFold(fold_ind)
      fold_test = dataset.getTestFold(fold_ind)
      self._train(fold_train, None, k_list)
      score.addFold(fold_test.getSales(), self.predict(fold_test))

    for month_ind in range(12):
      cur_rmsle = score.getRMSLE(month_ind)
      rmsle_list.append(cur_rmsle)
      
      if cur_rmsle < self.best_rmsle_list[month_ind]:
        self.best_rmsle_list[month_ind] = cur_rmsle
        self.k_list[month_ind] = k_list[month_ind]

    # print "get_rmsle_list k_list: %s returns %s" % (k_list, rmsle_list)
    return rmsle_list

  def tenary_search(self, dataset, num_folds):
    left_k_list = (RidgeRegression.min_k_list * 2. + RidgeRegression.max_k_list) / 3.
    left_rmsle_list = self.get_rmsle_list(dataset, num_folds, left_k_list)
    # print "  left_k_list: %s" % left_k_list
    
    right_k_list = (RidgeRegression.min_k_list + RidgeRegression.max_k_list * 2.) / 3.
    right_rmsle_list = self.get_rmsle_list(dataset, num_folds, right_k_list)
    # print "  right_k_list: %s" % right_k_list

    for month_ind in range(12):
      if left_rmsle_list[month_ind] < right_rmsle_list[month_ind]:
        RidgeRegression.max_k_list[month_ind] = right_k_list[month_ind]
      else:
        RidgeRegression.min_k_list[month_ind] = left_k_list[month_ind]

    if self.debug:  
      print "  min_k_list: %s" % (RidgeRegression.min_k_list)
      print "  max_k_list: %s" % (RidgeRegression.max_k_list)
      print "  k_list: %s" % self.k_list
    
  def cross_validate(self, dataset, num_folds):
    dataset.createFolds(num_folds)

    best_rmsle_list = [float("inf")] * 12
    self.k_list = [0] * 12

    for i in range(RidgeRegression.num_iterations):
      print "Tenary search iteration: %d" % (i+1)
      self.tenary_search(dataset, num_folds)
        
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
