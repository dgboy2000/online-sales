import csv
import random
import numpy as np

class DataSet:
  def __init__(self, train_set_flag):
    self.data_perm_inds = None
    self.features = None
    self.labels = None
    self.num_features = None
    self.num_folds = None
    self.num_samples = None
    self.train_set_flag = train_set_flag
    
  def getFeatures(self):
    return self.features
    
  def getLabels(self):
    return self.labels
    
  def getNumFeatures(self):
    return self.num_features
    
  def getNumSamples(self):
    return self.num_samples
    
  def importData(self, filename):
    data_reader = csv.reader(open(filename, 'rb'))
    header = data_reader.next()
    num_features = len(header) - 1
    
    labels = []
    features = []
    
    if self.train_set_flag:
      for row in data_reader:
        labels.append(row[0])
        features.append(row[1:])
      self.labels = np.asarray(labels, dtype=np.float64)
    else:
      for row in data_reader:
        features.append(row)
      
    self.features = np.asarray(features, dtype=np.float64)
    self.num_samples, self.num_features = self.features.shape
    
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
    ds_train.labels = self.labels[fold_inds]
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
    ds_test.labels = self.labels[fold_inds]
    ds_test.num_samples, ds_test.num_features = ds_test.features.shape
    return ds_test
    
  def getTestFoldInds(self, fold):
    return range(fold * self.getNumSamples() / self.num_folds, (fold+1) * self.getNumSamples() / self.num_folds)
    
    
    
    

