from LearnerBase import LearnerBase
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy import linalg
import Score

class RandomForest(object):
  n_values = [60, 80, 100, 125, 150]
  split_values = [1, 2, 3, 4]
  
  def __init__(self, n_estimators=100, min_split=2, debug=False):
    self.debug = debug

    if debug:
      RandomForest.n_values = [10]
      RandomForest.split_values = [2]

    self.rf_list = None
    self.n_estimators_list = None
    self.min_samples_split_list = None

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

      if n_estimators_list:
        rf = RandomForestRegressor(n_estimators=n_estimators_list[month_ind], min_samples_split=min_samples_split_list[month_ind])
      else:
        rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_split=min_samples_split)

      rf.fit(dataset.getFeaturesForMonth(month_ind), month_sales)
      self.rf_list.append(rf)

  def cross_validate(self, dataset, num_folds):
    dataset.createFolds(num_folds)
    
    best_rmsle_list = [float("inf")] * 12
    self.n_estimators_list = [0] * 12
    self.min_samples_split_list = [0] * 12
    
    for n in RandomForest.n_values:
      for split in RandomForest.split_values:

        score = Score.Score()

        for fold_ind in range(num_folds):
          if self.debug:  
            print "Running random forest with n_estimators=%d, min_split=%d on fold %d of %d folds" %(n, split, fold_ind, num_folds)

          fold_train = dataset.getTrainFold(fold_ind)
          fold_test = dataset.getTestFold(fold_ind)
          self._train(fold_train, n, split)
          score.addFold(fold_test.getSales(), self.predict(fold_test))
        
        for month_ind in range(12):
          cur_rmsle = score.getRMSLE(month_ind)

          if cur_rmsle < best_rmsle_list[month_ind]:
            best_rmsle_list[month_ind] = cur_rmsle
            self.n_estimators_list[month_ind] = n
            self.min_samples_split_list[month_ind] = split
        
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

