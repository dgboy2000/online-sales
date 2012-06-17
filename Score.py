import numpy as np

class Score:
  def __init__(self, labels, predictions):
    self.labels = labels
    self.predictions = predictions
    
  def getLogLoss(self):
    N = len(self.labels)
    return ( self.labels.dot(np.log(self.predictions)) + (1-self.labels).dot(np.log(1-self.predictions)) ) / -N
    
    
    
    
    
    
    
    
    
    
    
    
    
    