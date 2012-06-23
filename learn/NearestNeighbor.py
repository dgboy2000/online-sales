from LearnerBase import LearnerBase
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import Score

class NearestNeighbor(object):
  k_values = [2,3,5,8,13,21]
  
  def __init__(self, debug=False):
    self.debug = debug
    self.k = None
    self.k_list = None
    self.knn_list = None
      
  def _train(self, dataset, k, k_list = None):
    if self.debug:
      print "Training NearestNeighbor with %d features..." %(dataset.getNumFeatures())
    self.knn_list = []

    for month_ind in range(12):
      month_features = dataset.getFeaturesForMonth(month_ind)

      if self.debug:
        print "Learning on month %d of 12 with %d samples..." %(month_ind+1, month_features.shape[0])

      if k_list is not None:
        k = k_list[month_ind]

      knn = KNeighborsRegressor(k, weights='uniform')
      knn.fit(month_features, dataset.getSalesForMonth(month_ind))
      self.knn_list.append(knn)
    
  def cross_validate(self, dataset, num_folds):
    dataset.createFolds(num_folds)
    
    best_rmsle_list = [float("inf")] * 12
    best_k_list = [None] * 12
    for k in NearestNeighbor.k_values:
      cur_score = Score.Score()
      
      for fold_ind in range(num_folds):
        fold_train = dataset.getTrainFold(fold_ind)
        fold_test = dataset.getTestFold(fold_ind)
        self._train(fold_train, k)
        cur_score.addFold(fold_test.getSales(), self.predict(fold_test))
        
      for month_ind in range(12):
        cur_rmsle = cur_score.getRMSLE(month_ind)
        if cur_rmsle < best_rmsle_list[month_ind]:
          best_rmsle_list[month_ind] = cur_rmsle
          best_k_list[month_ind] = k
          
    self.k_list = list(best_k_list)  
    if self.debug:
      print "Best k-values by month: %s" %str(self.k_list)
    
  def train(self, dataset):
    self._train(dataset, None, k_list=self.k_list)
    
  def predict(self, dataset):
    features = dataset.getFeatures()
    num_samples, num_features = features.shape

    predictions = np.zeros((num_samples, 12))
    for month_ind in range(12):
      predictions[:, month_ind] = self.knn_list[month_ind].predict(features)

    return predictions
    
LearnerBase.register(NearestNeighbor)
