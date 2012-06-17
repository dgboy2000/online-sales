import numpy as np
import params
import pickle
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
    learner_predictions = np.zeros((dataset.getNumSamples(), len(self.learners)))
    dataset.createFolds(num_folds)
    for fold_ind in range(num_folds):
      if self.debug:
        print "Training learners on fold %d of %d..." %(fold_ind+1, num_folds)
      fold_train = dataset.getTrainFold(fold_ind)
      fold_test = dataset.getTestFold(fold_ind)
      prediction_inds = dataset.getTestFoldInds(fold_ind)
      self._loadOrTrainLearners(fold_train.getFeatures(), fold_train.getSales(), extension="%dof%d" %(fold_ind+1, num_folds))
      for learner_ind,learner in enumerate(self.learners):
        learner_predictions[prediction_inds, learner_ind] = learner.predict(fold_test.getFeatures())
    return learner_predictions    

  def _loadOrCVLearner(self, learner, dataset, num_folds):
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
    
  def _loadOrTrainLearner(self, learner, features, sales, extension=None):
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
      learner.train(features, sales)
      pickle.dump(learner, open(fname, 'w'))

    return learner

  def _loadOrTrainLearners(self, features, sales, extension=None):
    """Train all learners on a speficied dataset"""
    for learner_ind, learner in enumerate(self.learners):
      self.learners[learner_ind] = self._loadOrTrainLearner(learner, features, sales, extension=extension)

  def _makeAllSums(self, total, num_elts, delta=0.01):
    """Return a list of all possible tuples of num_elts elements which sum to total,
    where the numbers go up in increments of delta."""
    tuples = []
    if total == 0:
      return [(0,) * num_elts]
    if num_elts == 2:
      for a in np.arange(0, total+delta, delta):
        tuples.append((a, total-a))
      return tuples
    for a in np.arange(0, total+delta, delta):
      for tup in self._makeAllSums(total-a, num_elts-1, delta=delta):
        tuples.append((a,)+tup)
    return tuples

  def _selectLearnerWeights(self, sales):
    """Grid search for the optimal weights on the params."""
    # TODO: this searches for weights over 2 models; search over more
    if self.debug:
      print "Optimizing ensemble weights..."
    assert self.num_learners == 2, "Can only weight 2 models at present"
    best_loss = float("inf")
    best_weights = None
    for tup in self._makeAllSums(1, 2, delta=0.01):
      combined_predictions = np.asarray(tup).dot(np.asarray((self.learner_predictions[:, 0], self.learner_predictions[:, 1])))
      cur_score = Score.Score(sales, combined_predictions)
      cur_loss = cur_score.getLogLoss()
      if cur_loss < best_loss:
        if self.debug:
          print "Achieved new best ensemble loss %f with weights %s" %(cur_loss, str(tup))
        best_loss = cur_loss
        best_weights = tup
      else:
        if self.debug:
          print "Non-optimal ensemble loss %f with weights %s" %(cur_loss, str(tup))
    self.weights = np.asarray(best_weights)
        
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
    self._loadOrTrainLearners(dataset.getFeatures(), dataset.getSales(), extension='full')
    
  def predict(self, dataset):
    probs = np.zeros(dataset.getNumSamples())
    for learner_ind,learner in enumerate(self.learners):
      probs += learner.predict(dataset.getFeatures()) * self.weights[learner_ind]
      
    return probs
  
  
  
  
  
  
  
  
  
  
  