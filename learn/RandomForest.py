from LearnerBase import LearnerBase
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy import linalg
import Score

class RandomForest(object):
  n_values = [60, 80, 100, 125, 150]
  split_values = [1, 2, 3, 4]
  
  def __init__(self, n_estimators=100, min_split=2, debug=False):
    self.rf = None
    self.debug = debug
    self.min_split = min_split
    self.n_estimators = n_estimators
    self.params = None
    
  def cross_validate(self, dataset, num_folds):
    dataset.createFolds(num_folds)
    best_params = None
    best_score = float("inf")
    for n in RandomForest.n_values:
      for split in RandomForest.split_values:
        if self.debug:  
          print "Running random forest with n_estimators=%d, min_split=%d on %d folds" %(n, split, num_folds)
          
        learner_predictions = np.zeros(dataset.getNumSamples())
        for fold_ind in range(num_folds):
          fold_train = dataset.getTrainFold(fold_ind)
          fold_test = dataset.getTestFold(fold_ind)
          self.rf = RandomForestClassifier(n_estimators=n, min_split=split)
          self.rf.fit(fold_train.getFeatures(), fold_train.getLabels())
          fold_score = Score.Score(fold_test.getLabels(), self.predict(fold_test.getFeatures()))
          prediction_inds = dataset.getTestFoldInds(fold_ind)
          learner_predictions[prediction_inds] = self.predict(fold_test.getFeatures())
          
        cur_score = Score.Score(dataset.getLabels(), learner_predictions).getLogLoss()
        if cur_score < best_score:
          if self.debug:
            print "Achieved new best score %f" %cur_score
          best_params = (n, split)
          best_score = cur_score
        
    self.n_estimators, self.min_split = best_params
        
  def train(self, features, labels):
    if self.debug:
      print "Training random forest with n_estimators=%d, min_split=%d" %(self.n_estimators, self.min_split)
    self.rf = RandomForestClassifier(n_estimators=self.n_estimators, min_split=self.min_split)
    self.rf.fit(features, labels)
    
  def predict(self, features):
    num_samples, num_features = features.shape
    probs = [prob[1] for prob in self.rf.predict_proba(features)]
    return np.minimum(np.maximum(probs, 0.01*np.ones(num_samples)), 0.99*np.ones(num_samples))

    
LearnerBase.register(RandomForest)
    
    
    
    
    
    
    
    
    
    
    
