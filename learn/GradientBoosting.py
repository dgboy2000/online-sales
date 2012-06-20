from LearnerBase import LearnerBase
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

class GradientBoosting(object):
  def __init__(self, debug=False):
    self.debug = debug
    self.regressor_list = None
    
  def cross_validate(self, dataset, num_folds):
    pass
    
  def train(self, dataset):
    if self.debug:
      print "Train SuportVectorMachines with %d features..." %(dataset.getNumFeatures())

    self.regressor_list = []
    
    for month_ind in range(12):
      month_features = dataset.getFeaturesForMonth(month_ind)
      if self.debug:
        num_samples = month_features.shape[0]
        print "Learning on month %d of 12 with %d samples..." %(month_ind+1, num_samples)

      regressor =GradientBoostingRegressor()
      regressor.fit(month_features, dataset.getSalesForMonth(month_ind))
      self.regressor_list.append(regressor)
    
  def predict(self, dataset):
    features = dataset.getFeatures()
    num_samples, num_features = features.shape

    predictions = np.zeros((num_samples, 12))
    for month_ind in range(12):
      predictions[:, month_ind] = self.regressor_list[month_ind].predict(features)

    return predictions
    
LearnerBase.register(GradientBoosting)
