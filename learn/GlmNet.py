from LearnerBase import LearnerBase
import numpy as np
from scipy import linalg
import Score
import math
from sklearn.linear_model import ElasticNet

class GlmNet(object):
  rho_values =  [.1, .2, .3, .4, .5, .6, .7, .8, .9]

  def __init__(self, debug=False):
    self.max_depth = 1
    self.max_width = 10
    self.min_alpha = 0.
    self.max_alpha = .2

    self.debug = debug
    self.rho_list = None
    self.alpha_list = None
    self.params = None
    self.best_rmsle_list = [float("inf")] * 12
    self.is_best = False
    self.num_prunes = 0
    self.num_iterations = 0
    
  def _train(self, dataset, alpha_list, rho_list):
    self.regressor_list = []
    for month_ind in range(12):
      regressor = ElasticNet(alpha=alpha_list[month_ind], rho=rho_list[month_ind])
      regressor.fit(dataset.getFeaturesForMonth(month_ind), dataset.getSalesForMonth(month_ind))
      self.regressor_list.append(regressor)

  def get_rmsle_list(self, dataset, num_folds, alpha_list, rho_list):
    self.num_iterations += 1

    rmsle_list = []
    
    score = Score.Score()
    for fold_ind in range(num_folds):
      fold_train = dataset.getTrainFold(fold_ind)
      fold_test = dataset.getTestFold(fold_ind)
      self._train(fold_train, alpha_list, rho_list)
      score.addFold(fold_test.getSales(), self.predict(fold_test))

    for month_ind in range(12):
      cur_rmsle = score.getRMSLE(month_ind)
      rmsle_list.append(cur_rmsle)
      
      if cur_rmsle < self.best_rmsle_list[month_ind]:
        self.is_best = True
        self.best_rmsle_list[month_ind] = cur_rmsle
        self.alpha_list[month_ind] = alpha_list[month_ind]
        self.rho_list[month_ind] = rho_list[month_ind]

    print "alpha_list: %s" % alpha_list
    print "best_rho_list: %s" % str(self.rho_list)
    print "best_alpha_list: %s" % str(self.alpha_list)
    return rmsle_list

  def is_good(self, orders):
    # print "  is_bad orders: %s" % (orders)
    
    for i in range(12):
      order = [orders[i][j][1] for j in range(len(orders[i]))]
      # print "order: %s" % (order)
      if order != sorted(order) and order != sorted(order, reverse=True):
        return True

    return False

  def search(self, dataset, num_folds, depth, width, min_alpha_list, max_alpha_list, rho_list):
    if depth <= 0:
      return

    print "search: depth %d width %d" % (depth, width)
    # print "  min_alpha_list: %s" % str(min_alpha_list)
    # print "  max_alpha_list: %s" % str(max_alpha_list)

    min_alpha_lists = []
    max_alpha_lists = []
    rmsle_lists = []

    for i in range(width):
      min_alpha_lists.append(min_alpha_list + (max_alpha_list - min_alpha_list) * (float(i) / width))
      max_alpha_lists.append(min_alpha_list + (max_alpha_list - min_alpha_list) * (float(i+1) / width))

    self.is_best = False
    for i in range(width):
      alpha_list = (min_alpha_lists[i] + max_alpha_lists[i]) / 2
      # print "    alpha_list: %s" % str (alpha_list)

      rmsle_list = self.get_rmsle_list(dataset, num_folds, alpha_list, rho_list)
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
          new_min_alpha_list = []
          new_max_alpha_list = []
          for month_ind in range(12):
            new_min_alpha_list.append(min_alpha_lists[orders[month_ind][i][1]][month_ind])
            new_max_alpha_list.append(max_alpha_lists[orders[month_ind][i][1]][month_ind])
          # print "  new_min_alpha_list: %s" % str(new_min_alpha_list)
          # print "  new_max_alpha_list: %s" % str(new_max_alpha_list)
          self.search(dataset, num_folds, depth-1, width, np.array(new_min_alpha_list), np.array(new_max_alpha_list))  
      else:
        self.num_prunes += math.pow(self.max_width, depth)
    print "num_iterations: %d" % self.num_iterations
    print "num_prunes: %d" % self.num_prunes


  def cross_validate(self, dataset, num_folds):
    self.rho_list = [0.1] * 12
    self.alpha_list = [7.3476104736328125, 6.8925933837890625, 8.2190093994140625, 8.2137908935546875, 8.2814483642578125, 7.9523162841796875, 7.9453582763671875, 6.6873321533203125, 7.2606353759765625, 7.0821990966796875, 7.1634979248046875, 8.4375]

    if 1:
      dataset.createFolds(num_folds)
      for rho in GlmNet.rho_values:
        print "rho: %f" % (rho)
        best_rmsle_list = [float("inf")] * 12
        min_alpha_list = np.array([self.min_alpha] * 12)
        max_alpha_list = np.array([self.max_alpha] * 12)
        rho_list = [rho] * 12
        self.search(dataset, num_folds, self.max_depth, self.max_width, min_alpha_list, max_alpha_list, rho_list)
        
  def train(self, dataset):
    if self.debug:
      print "Training elastic net..."
    self._train(dataset, self.alpha_list, self.rho_list)
    
  def predict(self, dataset):
    features = dataset.getFeatures()
    num_samples, num_features = features.shape

    predictions = np.zeros((num_samples, 12))
    for month_ind in range(12):
      predictions[:, month_ind] = self.regressor_list[month_ind].predict(features)

    return predictions
    
LearnerBase.register(GlmNet)
