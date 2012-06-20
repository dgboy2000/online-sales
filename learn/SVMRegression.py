from LearnerBase import LearnerBase
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from scipy import linalg
import Score
import svmlight

class SVMRegression(object):
  poly_degrees = [1, 2]
  C_vals = [10]
  
  def __init__(self, n_estimators=100, min_split=2, debug=False):
    self.debug = debug
    self.formatted_data = None
    self.min_split = min_split
    self.n_estimators = n_estimators
    self.params = None
    self.svm_list = None
    
    # Parameters
    self.C = None
    self.poly_degree = None

  def _train_with_values(self, dataset, poly_degree=2, C=100):
    self.svm_list = []
    for month_ind in range(12):
      self._format_training_data(dataset, month_ind)
      if self.debug:
        print "Learning on month %d of 12 with %d samples..." %(month_ind+1, len(self.formatted_data))
      
      self.svm_list.append( svmlight.learn(self.formatted_data, type='regression', kernel='polynomial', poly_degree=poly_degree, C=C, verbosity=0) )
  
  def _format_training_data(self, dataset, month_ind):
    """Convert data into svmlight format for training: [(output, [(feature, value), ...], query_id), ...]"""
    formatted_data = []
    month_features = dataset.getFeaturesForMonth(month_ind)
    month_sales = dataset.getSalesForMonth(month_ind)
    for prod_ind,sales in enumerate(month_sales):
      feature_list = [(feat_ind+1,feat_val) for feat_ind,feat_val in enumerate(month_features[prod_ind,:])]
      formatted_data.append((sales, feature_list, 1))
      
    self.formatted_data = formatted_data
    
  def _format_test_data(self, dataset):
    """Convert data into svmlight format for prediction: [(0, [(feature, value), ...], query_id), ...]"""
    formatted_data = []
    features = dataset.getFeatures()
    num_samples = dataset.getNumSamples()
    for prod_ind in range(num_samples):
      feature_list = [(feat_ind+1,feat_val) for feat_ind,feat_val in enumerate(features[prod_ind, :])]
      formatted_data.append((0, feature_list, 1))
      
    self.formatted_data = formatted_data
    
  def cross_validate(self, dataset, num_folds):
    dataset.createFolds(num_folds)
    best_params = None
    best_rmsle = float("inf")
    for C in SVMRegression.C_vals:
      for poly_degree in SVMRegression.poly_degrees:
        if self.debug:  
          print "Running SVM regression with C=%d, poly_degree=%d on %d folds" %(C, poly_degree, num_folds)

        cur_score = Score.Score()

        for fold_ind in range(num_folds):
          fold_train = dataset.getTrainFold(fold_ind)
          fold_test = dataset.getTestFold(fold_ind)
          self._train_with_values(fold_train, poly_degree=poly_degree, C=C)
          cur_score.addFold(fold_test.getSales(), self.predict(fold_test))
          
        cur_rmsle = cur_score.getRMSLE()
        if cur_rmsle < best_rmsle:
          if self.debug:
            print "Achieved new best score %f" %cur_rmsle
          best_params = (C, poly_degree)
          best_rmsle = cur_rmsle
        
    self.C, self.poly_degree = best_params
        
  def train(self, dataset):
    if self.debug:
      print "Training SVM regression with C=%d, poly_degree=%d" %(self.C, self.poly_degree)
    self._train_with_values(dataset, poly_degree=self.poly_degree, C=self.C)
    
  def predict(self, dataset):
    assert self.svm_list is not None
    
    self._format_test_data(dataset)
    num_samples = dataset.getNumSamples()
    num_features = dataset.getNumFeatures()

    predictions = np.zeros((num_samples, 12))

    for month_ind in range(12):
      # import pdb;pdb.set_trace()
      predictions[:, month_ind] = svmlight.classify(self.svm_list[month_ind], self.formatted_data)
    return predictions
    
    
LearnerBase.register(SVMRegression)
    
    
    
    
    
    
    
    
    
    
    
