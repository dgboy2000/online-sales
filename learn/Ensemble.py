import numpy as np
import os
import params
import pickle
from scipy import linalg
import Score

class Ensemble:
  def __init__(self, debug=False):
    self.debug = debug
    self.learners = []
    self.num_learners = 0
    self.learner_predictions = None
    self.weights = None
    
  def _crossValidateLearners(self, dataset, num_folds):
    for learner_ind,learner in enumerate(self.learners):
      self.learners[learner_ind] = self._loadOrCVLearner(learner, dataset, num_folds)
    
  def _getLearnerCVPredictions(self, dataset, num_folds):
    learner_predictions = np.zeros((len(self.learners), dataset.getNumSamples(), 12))
    dataset.createFolds(num_folds)
    for fold_ind in range(num_folds):
      if self.debug:
        print "Training learners on fold %d of %d..." %(fold_ind+1, num_folds)
      fold_train = dataset.getTrainFold(fold_ind)
      fold_test = dataset.getTestFold(fold_ind)
      prediction_inds = dataset.getTestFoldInds(fold_ind)
      self._loadOrTrainLearners(fold_train, extension="%dof%d" %(fold_ind+1, num_folds))
      for learner_ind,learner in enumerate(self.learners):
        learner_predictions[learner_ind, prediction_inds, :] = learner.predict(fold_test)
    return learner_predictions    

  def _loadOrCVLearner(self, learner, dataset, num_folds):
    if not os.path.exists('cache'):
      os.mkdir('cache')
    
    learner_type = type(learner).__name__
    fname = 'cache/%s.pickle' %learner_type

    try:
      if (learner_type not in params.FEATURE_CACHE) or not params.FEATURE_CACHE[learner_type]:
        raise Exception('Do not use cache.')
      f = open(fname, 'rb')
      learner = pickle.load(f)
      if params.DEBUG:
        print "Using cached %s..." %learner_type
    except:
      if params.DEBUG:
        print "Cross-validating %s..." %learner_type
      learner.cross_validate(dataset, num_folds)
      pickle.dump(learner, open(fname, 'w'))

    return learner
    
  def _loadOrTrainLearner(self, learner, dataset, extension=None):
    learner_type = type(learner).__name__
    fname = 'cache/%s' %learner_type
    if extension is not None:
      fname += ".%s" %extension
    fname += ".pickle"

    try:
      if (learner_type not in params.FEATURE_CACHE) or not params.FEATURE_CACHE[learner_type]:
        raise Exception('Do not use cache.')
      f = open(fname, 'rb')
      learner = pickle.load(f)
      if params.DEBUG:
        print "Using cached %s..." %fname
    except:
      if params.DEBUG:
        print "Training and dumping %s..." %fname
      learner.train(dataset)
      pickle.dump(learner, open(fname, 'w'))

    return learner

  def _loadOrTrainLearners(self, dataset, extension=None):
    """Train all learners on a speficied dataset"""
    for learner_ind, learner in enumerate(self.learners):
      self.learners[learner_ind] = self._loadOrTrainLearner(learner, dataset, extension=extension)

  def _selectLearnerWeights(self, sales):
    """
    Krishna's formula for the optimal Ensemble weights under RMS(L)E:
    n samples
    k learners
    y is n-vector of actual sales
    yh is n x k matrix of learner predictions
    alpha is k-vector of optimal learner weights in an ensemble, by RMSE
    
    alpha = (yh^T * yh)^{-1} * (yh^T * y)
    """
    if self.debug:
      print "Optimizing ensemble weights..."

    # best_loss = float("inf")
    # best_weights = None
    # for tup in self._makeAllSums(1, 2, delta=0.01):
    #   combined_predictions = np.asarray(tup).dot(np.asarray((self.learner_predictions[:, 0], self.learner_predictions[:, 1])))
    #   cur_score = Score.Score(sales, combined_predictions)
    #   cur_loss = cur_score.getLogLoss()
    #   if cur_loss < best_loss:
    #     if self.debug:
    #       print "Achieved new best ensemble loss %f with weights %s" %(cur_loss, str(tup))
    #     best_loss = cur_loss
    #     best_weights = tup
    #   else:
    #     if self.debug:
    #       print "Non-optimal ensemble loss %f with weights %s" %(cur_loss, str(tup))
    # self.weights = np.asarray(best_weights)
    
    # Reshape all predictions and sales into long array
    # Run Krishna's equation for the optimal weights
    # TODO: separate
    num_samples, num_months = sales.shape
    num_learners = len(self.learners)

    sales = np.reshape(sales, num_samples * num_months)
    learner_predictions = np.reshape(self.learner_predictions, (num_learners, num_samples * num_months))

    # Drop all predictions and sales where sales are NaN
    not_nan_inds = [ind for ind,val in enumerate(sales) if val > 0.1]
    not_nan_predictions = np.zeros((num_learners, len(not_nan_inds)))
    sales = sales[not_nan_inds]
    for learner_ind in range(num_learners):
      not_nan_predictions[learner_ind, :] = learner_predictions[learner_ind, not_nan_inds]
    
    self.weights, residues, rank, s = linalg.lstsq(not_nan_predictions.transpose(), sales)
        
  def addLearner(self, learner):
    self.learners.append(learner)
    self.num_learners += 1
    
  def train(self, dataset, num_folds):
    """Train the individual models and learn the ensemble weights."""
    if self.debug:
      print "Training ensemble..."
    self._crossValidateLearners(dataset, num_folds)
    self.learner_predictions = self._getLearnerCVPredictions(dataset, num_folds)
    self._selectLearnerWeights(dataset.getSales())
    if self.debug:
      print "Training all models on all data..."
    self._loadOrTrainLearners(dataset, extension='full')
    
  def predict(self, dataset):
    sales = np.zeros((dataset.getNumSamples(), 12))
    for learner_ind,learner in enumerate(self.learners):
      sales += learner.predict(dataset) * self.weights[learner_ind]
      
    return sales
  
  
  
  
  
  
  
  
  
  
  