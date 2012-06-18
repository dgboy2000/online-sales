from LearnerBase import LearnerBase
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy import linalg
import Score

class RandomForest(object):
  # n_values = [60, 80, 100, 125, 150]
  # split_values = [1, 2, 3, 4]
  n_values = [10]
  split_values = [2]
  
  def __init__(self, n_estimators=100, min_split=2, debug=False):
    self.rf_list = None
    self.debug = debug
    self.min_split = min_split
    self.n_estimators = n_estimators
    self.params = None


  def _train_with_values(self, dataset, n_estimators, min_samples_split):
    self.rf_list = []
    for month_ind in range(12):
      month_sales = dataset.getSalesForMonth(month_ind)
      if self.debug:
        print "Learning on month %d of 12 with %d samples..." %(month_ind+1, len(month_sales))

      rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_split=min_samples_split)
      rf.fit(dataset.getFeaturesForMonth(month_ind), month_sales)
      self.rf_list.append(rf)

  def cross_validate(self, dataset, num_folds):
    dataset.createFolds(num_folds)
    best_params = None
    best_rmsle = float("inf")
    for n in RandomForest.n_values:
      for split in RandomForest.split_values:
        if self.debug:  
          print "Running random forest with n_estimators=%d, min_split=%d on %d folds" %(n, split, num_folds)

        cur_score = Score.Score()

        for fold_ind in range(num_folds):
          fold_train = dataset.getTrainFold(fold_ind)
          fold_test = dataset.getTestFold(fold_ind)
          self._train_with_values(fold_train, n, split)
          cur_score.addFold(fold_test.getSales(), self.predict(fold_test))
          
        cur_rmsle = cur_score.getRMSLE()
        if cur_rmsle < best_rmsle:
          if self.debug:
            print "Achieved new best score %f" %cur_rmsle
          best_params = (n, split)
          best_rmsle = cur_rmsle
        
    self.n_estimators, self.min_split = best_params
        
  def train(self, dataset):
    if self.debug:
      print "Training random forest with n_estimators=%d, min_split=%d" %(self.n_estimators, self.min_split)
    self._train_with_values(dataset, self.n_estimators, self.min_split)
    
  def predict(self, dataset):
    features = dataset.getFeatures()
    num_samples, num_features = features.shape

    predictions = np.zeros((num_samples, 12))

    for month_ind in range(12):
      predictions[:, month_ind] = self.rf_list[month_ind].predict(features)
    return predictions
    
LearnerBase.register(RandomForest)
    
    
    
    
    
    
    
    
    
    
    
