import copy
import csv
import math
import numpy as np
import params
import random


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
    self.quant_means = None
    self.quant_variances = None
    self.train_set_flag = train_set_flag
    self.useless_features = set()
    
  def _getQuantitativeHeaderInds(self):
    quantitative_header_inds = []
    for ind,header in enumerate(self.headers):
      if header[:4].lower() == 'quan':
        quantitative_header_inds.append(ind)
    return quantitative_header_inds
    
  def getHeaders(self):
    return self.headers
    
  def getFeatures(self):
    return self.features
    
  def getQuantitativeFeatures(self):
    return self.features[:, self._getQuantitativeHeaderInds()]
    
  def getSales(self):
    return self.sales    
    

  def getIndsForMonth(self, month_ind):
    """Return the indices of the non-NaN sales for the specified month ind [0-11]"""
    return [ind for ind,val in enumerate(self.sales[:, month_ind]) if val > 0.0]
    
  def getFeaturesForMonth(self, month_ind):
    """Return the features with non-NaN sales for the specified month ind [0-11]."""
    month_inds = self.getIndsForMonth(month_ind)
    return self.features[month_inds, :]
    
  def getQuantitativeFeaturesForMonth(self, month_ind):
    return self.getFeaturesForMonth(month_ind)[:, self._getQuantitativeHeaderInds()]
    
  def getSalesForMonth(self, month_ind):
    """Return the non-NaN sales data for the specified month ind [0-11]."""
    month_inds = self.getIndsForMonth(month_ind)
    return self.sales[self.getIndsForMonth(month_ind), month_ind]
    
    
  def getNumFeatures(self):
    return self.num_features
    
  def getNumQuantitativeFeatures(self):
    return len(self._getQuantitativeHeaderInds())
    
  def getNumSamples(self):
    return self.num_samples
    
  def logTransformQuantitativeFeatures(self):
    quant_feat_inds = self._getQuantitativeHeaderInds()
    if (self.features[:, quant_feat_inds] == 0).sum() > 0:
      ZERO_VAL = 0.2
      print "WARNING: some quantitative features have value 0; changing these to %f to take log" %ZERO_VAL
      self.features[:, quant_feat_inds] = np.max(np.dstack((self.features[:, quant_feat_inds], ZERO_VAL * np.ones((self.getNumSamples(), len(quant_feat_inds))) )), axis=2)
    self.features[:, quant_feat_inds] = np.log(self.features[:, quant_feat_inds])
    
  def standardizeQuantitativeFeatures(self, means=None, variances=None):
    """Standardize the quantitative features to be mean 0, variance 1. If means/variances are specified,
    use them to normalize the data."""
    quant_feat_inds = self._getQuantitativeHeaderInds()
    if means is None:
      means = np.mean(self.features[:, quant_feat_inds], axis=0)
    if variances is None:
      variances = np.var(self.features[:, quant_feat_inds], axis=0)
    self.quant_means = means
    self.quant_variances = variances
    for quant_ind,feat_ind in enumerate(quant_feat_inds):
      self.features[:, feat_ind] = (self.features[:, feat_ind] - means[quant_ind]) / np.sqrt(variances[quant_ind])
      
  def getQuantitativeFeatureMeans(self):
    return self.quant_means
    
  def getQuantitativeFeatureVariances(self):
    return self.quant_variances

  def _normalizeSales(self, row):
    ret = copy.deepcopy(row);
    for i in range(len(ret)):
      if math.isnan(float(ret[i])):
        # Hardcoding nan to 0.
        ret[i] = 0
      else:
        ret[i] = math.log(float(ret[i]) + 1)
    return ret
    
  def _changeNanToZero(self):
    for i in range(self.num_samples):
      for j in range(self.num_features):
        if math.isnan(self.features[i,j]):
          self.features[i,j] = 0.0
          
  def _addDerivedFeatures(self):
    features = self.features
    headers = self.headers
    num_samples = self.getNumSamples()
    
    date1_ind = headers.index('Date_1')
    date2_ind = headers.index('Date_2')
    
    headers.append('Quant_PrelaunchMarketingDays')
    new_feat = (features[:, date1_ind]-features[:, date2_ind]).reshape((num_samples, 1))
    features = np.append(features, new_feat, axis=1)
    
    headers.append('Quant_LaunchDayOfYear')
    new_feat = np.mod((features[:, date1_ind]).reshape((num_samples, 1)), 365)
    features = np.append(features, new_feat, axis=1)
    
    headers.append('Quant_LaunchYear')
    new_feat = np.floor(np.divide((features[:, date1_ind]).reshape((num_samples, 1)), 365))
    features = np.append(features, new_feat, axis=1)

    headers.append('Quant_LaunchMonth')
    new_feat = np.floor(np.divide(np.mod((features[:, date1_ind]).reshape((num_samples, 1)), 365), 30))
    features = np.append(features, new_feat, axis=1)
    
    self._setFeatures(features, headers)
    
  def getUselessFeatures(self):
    return self.useless_features
  def dropUselessFeatures(self, feature_inds):
    if len(self.useless_features) > 0:
      raise "Already processed useful features; can't do this twice"
      
    self.useless_features = set(feature_inds)
    self._dropFeatureInds(self.useless_features)
    
  def _dropFeatureInds(self, feature_inds):
    useful_features = np.zeros((self.num_samples, self.num_features - len(feature_inds)))
    useful_feat_ind = 0
    useful_headers = []
    for feat_ind in range(self.num_features):
      if feat_ind not in feature_inds:
        useful_features[:, useful_feat_ind] = self.features[:, feat_ind]
        useful_feat_ind += 1
        useful_headers.append(self.headers[feat_ind])
    
    self._setFeatures(useful_features, useful_headers)
    
  def _detectUsefulFeatures(self):
    if len(self.useless_features) > 0:
      raise "Already processed useful features; can't do this twice"
      
    self.useless_features.update(self._detectZeroVarianceFeatures())
    self.useless_features.update(self._detectDuplicateFeatures())
    self._dropFeatureInds(self.useless_features)
    
  def _detectZeroVarianceFeatures(self):
    zero_variance_features = []
    for feat_ind in range(self.num_features):
      if np.var(self.features[:, feat_ind]) == 0:
        zero_variance_features.append(feat_ind)
    if params.DEBUG:
      print "Found %d zero-variance features: %s" %(len(zero_variance_features), str(zero_variance_features))
    return zero_variance_features
    
  def _detectDuplicateFeatures(self):
    duplicate_feature_to_orig = {}
    for feat_ind in range(self.num_features):
      for other_feat_ind in range(feat_ind):
        if np.var(self.features[:, feat_ind] - self.features[:, other_feat_ind]) == 0:
          duplicate_feature_to_orig[feat_ind] = other_feat_ind
          break
    if params.DEBUG:
      print "Found %d duplicate features: %s" %(len(duplicate_feature_to_orig), str(duplicate_feature_to_orig))
    return duplicate_feature_to_orig.keys()

  def _setFeatures(self, features, headers):
    self.features = np.asarray(features, dtype=np.float64)
    self.num_samples, self.num_features = self.features.shape
    self.headers = headers

  def importData(self, filename, ):
    self.headers = None
    self.ids = None
    self.sales = None
    self.features = None
    
    data_reader = csv.reader(open(filename, 'rb'))
    headers = data_reader.next()
    num_features = len(headers) - 12
    
    ids = []
    sales = []
    features = []
    
    if self.train_set_flag:
      for row in data_reader:
        sales.append(self._normalizeSales(row[:12]))
        features.append(row[12:])
      self.sales = np.asarray(sales, dtype=np.float64)
      headers = headers[12:]
    else:
      for row in data_reader:
        ids.append(int(row[0]))
        features.append(row[1:])
      self.ids = np.asarray(ids, dtype=np.int32)
      headers = headers[1:]
    
    self._setFeatures(features, headers)
    
    self._changeNanToZero()
    self._addDerivedFeatures()
          
    if self.train_set_flag:
      self._detectUsefulFeatures()
    
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
    return self._getFold(fold_inds)
    
  def getTrainFoldInds(self, fold):
    return range(0, fold * self.getNumSamples() / self.num_folds) + range((fold+1) * self.getNumSamples() / self.num_folds, self.getNumSamples())
    
  def getTestFold(self, fold):
    if fold < 0 or fold >= self.num_folds:
      raise Exception("Requested invalid test fold %d; there are %d total folds" %(fold, self.num_folds))
    fold_inds = self.getTestFoldInds(fold)
    return self._getFold(fold_inds)
    
  def _getFold(self, fold_inds):
    fold = DataSet(self.train_set_flag)
    fold.features = self.features[fold_inds, :]
    fold.sales = self.sales[fold_inds, :]
    fold.headers = list(self.headers)
    fold.num_samples, fold.num_features = fold.features.shape
    return fold
    
    
  def getTestFoldInds(self, fold):
    return range(fold * self.getNumSamples() / self.num_folds, (fold+1) * self.getNumSamples() / self.num_folds)
    
    
    
    

