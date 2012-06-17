import csv
import random
import math
import numpy as np

class DataSet:
  def __init__(self, train_set_flag):
    self.data_perm_inds = None
    self.headers = None
    self.ids = None
    self.features = None
    self.sales = None
    self.num_features = None
    self.num_folds = None
    self.num_samples = None
    self.train_set_flag = train_set_flag
    
  def getFeatures(self):
    return self.features
    
  def getSales(self):
    return self.sales    
    

  def getIndsForMonth(self, month_ind):
    """Return the indices of the non-NaN sales for the specified month ind [0-11]"""
    return [ind for ind,val in enumerate(self.sales[:, month_ind]) if val > 1.0]
    
  def getFeaturesForMonth(self, month_ind):
    """Return the features with non-NaN sales for the specified month ind [0-11]."""
    return self.features[self.getIndsForMonth(month_ind), :]
    
  def getSalesForMonth(self, month_ind):
    """Return the non-NaN sales data for the specified month ind [0-11]."""
    return self.sales[self.getIndsForMonth(month_ind), month_ind]
    
    
  def getNumFeatures(self):
    return self.num_features
    
  def getNumSamples(self):
    return self.num_samples
    
  def importData(self, filename):
    self.headers = None
    self.ids = None
    self.sales = None
    self.features = None
    
    data_reader = csv.reader(open(filename, 'rb'))
    self.headers = data_reader.next()
    num_features = len(self.headers) - 12
    
    ids = []
    sales = []
    features = []
    
    if self.train_set_flag:
      for row in data_reader:
        sales.append(row[:12])
        features.append(row[12:])
      self.sales = np.asarray(sales, dtype=np.float64)
    else:
      for row in data_reader:
        ids.append(row[0])
        features.append(row[1:])
      self.ids = np.asarray(ids, dtype=np.float64)
      
    self.features = np.asarray(features, dtype=np.float64)
    self.num_samples, self.num_features = self.features.shape
    
    # Hack to make things work
    for i in range(self.num_samples):
      for j in range(self.num_features):
        if math.isnan(self.features[i,j]):
          self.features[i,j] = 0.0
      if self.train_set_flag:
        for j in range(12):
          if math.isnan(self.sales[i,j]):
            self.sales[i,j] = 0.0
    
  def createFolds(self, num_folds):
    assert self.train_set_flag, "Can only create folds of training data"
    self.num_folds = num_folds
    # # TODO: should we randomize the indices as follows?
    # self.data_perm_inds = range(self.getNumSamples())
    # random.shuffle(self.data_perm_inds)
    
  def getTrainFold(self, fold):
    if fold < 0 or fold >= self.num_folds:
      raise Exception("Requested invalid train fold %d; there are %d total folds" %(fold, self.num_folds))
    fold_inds = self.getTrainFoldInds(fold)
    ds_train = DataSet(self.train_set_flag)
    ds_train.features = self.features[fold_inds, :]
    ds_train.sales = self.sales[fold_inds]
    ds_train.num_samples, ds_train.num_features = ds_train.features.shape
    return ds_train
    
  def getTrainFoldInds(self, fold):
    return range(0, fold * self.getNumSamples() / self.num_folds) + range((fold+1) * self.getNumSamples() / self.num_folds, self.getNumSamples())
    
  def getTestFold(self, fold):
    if fold < 0 or fold >= self.num_folds:
      raise Exception("Requested invalid test fold %d; there are %d total folds" %(fold, self.num_folds))
    fold_inds = self.getTestFoldInds(fold)
    ds_test = DataSet(self.train_set_flag)
    ds_test.features = self.features[fold_inds, :]
    ds_test.sales = self.sales[fold_inds]
    ds_test.num_samples, ds_test.num_features = ds_test.features.shape
    return ds_test
    
  def getTestFoldInds(self, fold):
    return range(fold * self.getNumSamples() / self.num_folds, (fold+1) * self.getNumSamples() / self.num_folds)
    
    
    
    

