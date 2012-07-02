from LearnerBase import LearnerBase
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import Score

class GradientBoosting(object):
  def __init__(self, debug=False):
    self.debug = debug
    self.max_depth = 1
    self.max_width = 5
    self.min_n_estimators = 50
    self.max_n_estimators = 250
    self.num_iterations = 0
    self.num_prunes = 0
    self.is_best = False

    self.best_rmsle_list = None
    self.regressor_list = None
    self.n_estimators_list = None

  def _train(self, dataset, n_estimators_list):
    self.regressor_list = []

    for month_ind in range(12):
      month_sales = dataset.getSalesForMonth(month_ind)
      regressor = GradientBoostingRegressor(n_estimators=n_estimators_list[month_ind])
      regressor.fit(dataset.getFeaturesForMonth(month_ind), month_sales)
      self.regressor_list.append(regressor)

  def get_rmsle_list(self, dataset, num_folds, n_estimators_list):
    self.num_iterations += 1
    print 'n_estimators_list: %s' % n_estimators_list
    rmsle_list = []
    score = Score.Score()
  
    for fold_ind in range(num_folds):
      print '  fold_ind: %d' % fold_ind
      fold_train = dataset.getTrainFold(fold_ind)
      fold_test = dataset.getTestFold(fold_ind)
      self._train(fold_train, n_estimators_list)
      score.addFold(fold_test.getSales(), self.predict(fold_test))
        
      for month_ind in range(12):
        cur_rmsle = score.getRMSLE(month_ind)
        rmsle_list.append(cur_rmsle)
          
        if cur_rmsle < self.best_rmsle_list[month_ind]:
          self.is_best = True
          self.best_rmsle_list[month_ind] = cur_rmsle
          self.n_estimators_list[month_ind] = n_estimators_list[month_ind]

    print 'best_n_estimators_list: %s' % self.n_estimators_list
    return rmsle_list

  def is_good(self, orders):
    # print "  is_bad orders: %s" % (orders)
    
    for i in range(12):
      order = [orders[i][j][1] for j in range(len(orders[i]))]
      # print "order: %s" % (order)
      if order != sorted(order) and order != sorted(order, reverse=True):
        return True

    return False

  def search(self, dataset, num_folds, depth, width, min_n_estimators_list, max_n_estimators_list):
    if depth <= 0:
      return

    print "search: depth %d width %d" % (depth, width)
    # print "min_n_estimators_list: %s" % min_n_estimators_list
    # print "max_n_estimators_list: %s" % max_n_estimators_list

    min_n_estimators_lists = []
    max_n_estimators_lists = []
    rmsle_lists = []

    if max_n_estimators_list[0] - min_n_estimators_list[0] <= width:
      for i in range(width):
        min_n_estimators_lists.append(min_n_estimators_list + np.array([i] * 12))
        max_n_estimators_lists.append(min_n_estimators_list + np.array([i+1] * 12))
    else:
      for i in range(width):
        min_n_estimators_lists.append(min_n_estimators_list + (max_n_estimators_list - min_n_estimators_list) * (i) / width)
        max_n_estimators_lists.append(min_n_estimators_list + (max_n_estimators_list - min_n_estimators_list) * (i+1) / width)

    self.is_best = False
    for i in range(width):
      n_estimators_list = (min_n_estimators_lists[i] + max_n_estimators_lists[i]) / 2
      rmsle_list = self.get_rmsle_list(dataset, num_folds, n_estimators_list)
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
          new_min_n_estimators_list = []
          new_max_n_estimators_list = []

          for month_ind in range(12):
            new_min_n_estimators_list.append(min_n_estimators_lists[orders[month_ind][i][1]][month_ind])
            new_max_n_estimators_list.append(max_n_estimators_lists[orders[month_ind][i][1]][month_ind])
          # print "  new_min_k_list: %s" % str(new_min_k_list)
          # print "  new_max_k_list: %s" % str(new_max_k_list)
          self.search(dataset, num_folds, depth-1, width, np.array(new_min_n_estimators_list), np.array(new_max_n_estimators_list))
      else:
        self.num_prunes += math.pow(self.max_width, depth)
    print "num_iterations: %d" % self.num_iterations
    print "num_prunes: %d" % self.num_prunes

  def cross_validate(self, dataset, num_folds):
    self.n_estimators_list = [103, 232, 115, 126, 117, 185, 134, 117, 101, 204, 179, 85]

    if 1:
      dataset.createFolds(num_folds)
      self.best_rmsle_list = [float("inf")] * 12
      
      self.min_n_estimators_list = np.array([self.min_n_estimators] * 12)
      self.max_n_estimators_list = np.array([self.max_n_estimators] * 12)
      self.search(dataset, num_folds, self.max_depth, self.max_width, self.min_n_estimators_list, self.max_n_estimators_list)
    
  def train(self, dataset):
    if self.debug:
      print "Training GradientBoosting model with %d features..." %(dataset.getNumFeatures())
    self._train(dataset, self.n_estimators_list)
    
  def predict(self, dataset):
    features = dataset.getFeatures()
    num_samples, num_features = features.shape

    predictions = np.zeros((num_samples, 12))
    for month_ind in range(12):
      predictions[:, month_ind] = self.regressor_list[month_ind].predict(features)

    return predictions
    
LearnerBase.register(GradientBoosting)
