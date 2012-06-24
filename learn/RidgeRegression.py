from LearnerBase import LearnerBase
import numpy as np
from scipy import linalg
import Score
import math

class RidgeRegression(object):
  def __init__(self, debug=False):
    self.debug = debug
    self.k_list = None
    self.params = None
    self.best_rmsle_list = [float("inf")] * 12

    self.max_depth = 5
    self.max_width = 8

    self.is_best = False
    self.num_prunes = 0
    self.num_iterations = 0
    
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
    self.num_iterations += 1

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
        self.is_best = True
        self.best_rmsle_list[month_ind] = cur_rmsle
        self.k_list[month_ind] = k_list[month_ind]
      # print "best_rmsle_list: %s" % str(self.best_rmsle_list)
    print "k_list: %s" % k_list
    print "best_k_list: %s" % str(self.k_list)
    return rmsle_list

  def is_good(self, orders):
    # print "  is_bad orders: %s" % (orders)
    
    for i in range(12):
      order = [orders[i][j][1] for j in range(len(orders[i]))]
      # print "order: %s" % (order)
      if order != sorted(order) and order != sorted(order, reverse=True):
        return True

    return False

  def search(self, dataset, num_folds, depth, width, min_k_list, max_k_list):
    if depth <= 0:
      return

    print "search: depth %d width %d" % (depth, width)
    # print "  min_k_list: %s" % str(min_k_list)
    # print "  max_k_list: %s" % str(max_k_list)

    min_k_lists = []
    max_k_lists = []
    rmsle_lists = []

    for i in range(width):
      min_k_lists.append(min_k_list + (max_k_list - min_k_list) * (float(i) / width))
      max_k_lists.append(min_k_list + (max_k_list - min_k_list) * (float(i+1) / width))

    self.is_best = False
    for i in range(width):
      k_list = (min_k_lists[i] + max_k_lists[i]) / 2
      # print "    k_list: %s" % str (k_list)

      rmsle_list = self.get_rmsle_list(dataset, num_folds, k_list)
      # print "    rmsle_list: %s" % str (rmsle_list)
      rmsle_lists.append(rmsle_list)

    if depth > 1:
      orders = []

      for month_ind in range(12):
        order = []
        for i in range(width):
          order.append((rmsle_lists[i][month_ind], i))
        # print 'order: %s' % str(order)
        order.sort()
        # print 'orderred: %s' % str(order)
        orders.append(order)

      if self.is_best or self.is_good(orders):
        for i in range(width):
          # print "range: %d" % i
          new_min_k_list = []
          new_max_k_list = []
          for month_ind in range(12):
            new_min_k_list.append(min_k_lists[orders[month_ind][i][1]][month_ind])
            new_max_k_list.append(max_k_lists[orders[month_ind][i][1]][month_ind])
          # print "  new_min_k_list: %s" % str(new_min_k_list)
          # print "  new_max_k_list: %s" % str(new_max_k_list)
          self.search(dataset, num_folds, depth-1, width, np.array(new_min_k_list), np.array(new_max_k_list))  
      else:
        self.num_prunes += math.pow(self.max_width, depth)
    print "num_iterations: %d" % self.num_iterations
    print "num_prunes: %d" % self.num_prunes


  def cross_validate(self, dataset, num_folds):
    self.k_list = [7.3476104736328125, 6.8925933837890625, 8.2190093994140625, 8.2137908935546875, 8.2814483642578125, 7.9523162841796875, 7.9453582763671875, 6.6873321533203125, 7.2606353759765625, 7.0821990966796875, 7.1634979248046875, 8.4375]
    if 0:
      dataset.createFolds(num_folds)
      best_rmsle_list = [float("inf")] * 12
      min_k_list = np.array([6] * 12)
      max_k_list = np.array([9] * 12)
      self.search(dataset, num_folds, self.max_depth, self.max_width, min_k_list, max_k_list)
        
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
