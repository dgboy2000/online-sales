from LearnerBase import LearnerBase
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy import linalg
import Score

class RandomForest(object):
  split_values = [1, 2, 3, 4]
  
  def __init__(self, debug=False):
    self.debug = debug

    self.rf_list = None
    self.n_estimators_list = None
    self.min_samples_split_list = None
    self.best_rmsle_list = None

  def _train(self, dataset, 
             n_estimators, 
             min_samples_split,
             n_estimators_list = None,
             min_samples_split_list = None):
    self.rf_list = []

    for month_ind in range(12):
      month_sales = dataset.getSalesForMonth(month_ind)

      if self.debug:
        print "Learning on month %d of 12 with %d samples..." %(month_ind+1, len(month_sales))

      if n_estimators_list is not None:
        n_estimators = n_estimators_list[month_ind]

      if min_samples_split_list is not None:
        min_samples_split = min_samples_split_list[month_ind]

      rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_split=min_samples_split)
      rf.fit(dataset.getFeaturesForMonth(month_ind), month_sales)
      self.rf_list.append(rf)

  def get_rmsle_list(self, dataset, num_folds, split, n_list):
    rmsle_list = []
    score = Score.Score()
  
    for fold_ind in range(num_folds):
      if self.debug:  
        print "    Running random forest on fold %d of %d folds" %(fold_ind, num_folds)

      fold_train = dataset.getTrainFold(fold_ind)
      fold_test = dataset.getTestFold(fold_ind)
      self._train(fold_train, None, split, n_list)
      score.addFold(fold_test.getSales(), self.predict(fold_test))
        
      for month_ind in range(12):
        cur_rmsle = score.getRMSLE(month_ind)
        rmsle_list.append(cur_rmsle)
          
        if cur_rmsle < self.best_rmsle_list[month_ind]:
          self.best_rmsle_list[month_ind] = cur_rmsle
          self.n_estimators_list[month_ind] = n_list[month_ind]
          self.min_samples_split_list[month_ind] = split
    return rmsle_list

  def tenary_search(self, dataset, num_folds, split):
    left_n_list = (RandomForest.min_n_list * 2 + RandomForest.max_n_list) / 3
    print "    left_n_list: %s" % left_n_list
    left_rmsle_list = self.get_rmsle_list(dataset, num_folds, split, left_n_list)
    
    right_n_list = (RandomForest.min_n_list + RandomForest.max_n_list * 2) / 3
    print "    right_n_list: %s" % right_n_list
    right_rmsle_list = self.get_rmsle_list(dataset, num_folds, split, right_n_list)

    print "    left_rmsle_list: %s" % left_rmsle_list
    print "    right_rmsle_list: %s" % right_rmsle_list

    for month_ind in range(12):
      if left_rmsle_list[month_ind] < right_rmsle_list[month_ind]:
        RandomForest.max_n_list[month_ind] = right_n_list[month_ind] - 1
      else:
        RandomForest.min_n_list[month_ind] = left_n_list[month_ind] + 1

    if self.debug:  
      print "    min_n_list: %s" % (RandomForest.min_n_list)
      print "    max_n_list: %s" % (RandomForest.max_n_list)
      print "    n_list: %s" % self.n_estimators_list

  def cross_validate(self, dataset, num_folds):
    dataset.createFolds(num_folds)
    self.best_rmsle_list = [float("inf")] * 12
    self.n_estimators_list = [0] * 12
    self.min_samples_split_list = [0] * 12

    if self.debug:
      self.n_estimators_list = [100, 106, 117, 84, 91, 50, 100, 66, 58, 33, 77, 117]
      self.min_samples_split_list = [2, 1, 2, 4, 1, 1, 2, 1, 1, 2, 3, 1]
    else:
      for split in RandomForest.split_values:
        print "Split: %d" % (split)
        RandomForest.num_iterations = 10
        RandomForest.min_n_list = np.array([1] * 12)
        RandomForest.max_n_list = np.array([150] * 12)

        for i in range(RandomForest.num_iterations):
          print "  Tenary search iteration: %d" % (i+1)
          self.tenary_search(dataset, num_folds, split)
        
  def train(self, dataset):
    if self.debug:
      print "Training random forest with n_estimators_list: %s, min_samples_split_list: %s" %(str(self.n_estimators_list), str(self.min_samples_split_list))
    self._train(dataset, None, None, self.n_estimators_list, self.min_samples_split_list)
    
  def predict(self, dataset):
    features = dataset.getFeatures()
    num_samples, num_features = features.shape

    predictions = np.zeros((num_samples, 12))

    for month_ind in range(12):
      predictions[:, month_ind] = self.rf_list[month_ind].predict(features)
    return predictions
    
LearnerBase.register(RandomForest)

