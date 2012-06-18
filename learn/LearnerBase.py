import abc

class LearnerBase(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def cross_validate(self, dataset, num_folds):
    """Choose and save the best parameters by cross-validation."""
    return
    
  @abc.abstractmethod
  def train(self, dataset):
    """Train the learner on the specified dataset, assuming CV has already occurred."""
    return

  @abc.abstractmethod
  def predict(self, dataset):
    """Return a real-number prediction for each sales month on the specified dataset."""
    return
   


















