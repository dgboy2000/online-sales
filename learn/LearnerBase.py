import abc

class LearnerBase(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def cross_validate(self, dataset, num_folds):
    """Choose and save the best parameters by cross-validation."""
    return
    
  @abc.abstractmethod
  def train(self, features, sales):
    """Train the learner on the specified features and sales data."""
    return

  @abc.abstractmethod
  def predict(self, features):
    """Return a real-number prediction for each feature vector in the specified array."""
    return
   


















